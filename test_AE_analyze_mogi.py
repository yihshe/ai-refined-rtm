import argparse
import json
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import pandas as pd
import numpy as np
from rtm_torch.rtm import RTM


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    # NOTE the test set needs to be set beforehand e.g. in dataset.py
    data_loader = getattr(module_data, config['data_loader']['type_test'])(
        config['data_loader']['data_dir_test'],
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        # training=False,
        num_workers=2
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss_test'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    data_key = config['trainer']['input_key']
    target_key = config['trainer']['output_key']

    # analyze the reconstruction loss per band
    if config['arch']['type'] == 'VanillaAE':
        # ATTRS = ['1', '2', '3', '4', '5']
        ATTRS = [str(i+1) for i in range(config['arch']['args']['hidden_dim'])]
    else:
        # ATTRS = ['xcen', 'ycen', 'd', 'dV_factor', 'dV_power', 'dV']
        ATTRS = ['xcen', 'ycen', 'd', 'dV']

    analyzer = {}
    mogi = RTM()
    MEAN = np.load('/maps/ys611/ai-refined-rtm/data/mogi/train_x_mean.npy')
    SCALE = np.load('/maps/ys611/ai-refined-rtm/data/mogi/train_x_scale.npy')
    station_info = json.load(open(
        '/maps/ys611/ai-refined-rtm/configs/mogi/station_info.json'))
    GPS = []
    for direction in ['ux', 'uy', 'uz']:
        for station in station_info.keys():
            GPS.append(f'{direction}_{station}')

    with torch.no_grad():
        for batch_idx, data_dict in enumerate(data_loader):
            data = data_dict[data_key].to(device)
            target = data_dict[target_key].to(device)
            if config['arch']['type'] in ['AE_Mogi', 'AE_Mogi_corr']:
                outputs = model(data)
                output = outputs[-1]
                latent = outputs[1]
                if config['arch']['type'] == 'AE_Mogi_corr':
                    init_output = outputs[2]
                    bias = output - init_output
            elif config['arch']['type'] == 'VanillaAE':
                output = model(data)
                latent = model.encode(data)

            # calcualte the corrected bias if the model is AE_RTM_corr
            if config['arch']['type'] in ['AE_Mogi_corr']:
                # calculate the direct output from RTM
                # output, init_output = model.decode(latent)

                # calculate the bias
                # NOTE bias in original scale = bias*SCALE if the data is scaled
                # bias = output - init_output
                data_concat(analyzer, 'init_output', init_output)
                data_concat(analyzer, 'bias', bias)

            if config['arch']['type'] == 'VanillaAE':
                assert len(
                    ATTRS) == latent.shape[1], "latent shape does not match"
            else:
                assert ATTRS == list(latent.keys()), "latent keys do not match"
                # latent is a dictionary of parameters, convert it to a tensor
                latent = torch.stack([latent[k] for k in latent.keys()], dim=1)

            # l2_per_band = torch.square(output-target)
            data_concat(analyzer, 'output', output)
            data_concat(analyzer, 'target', target)
            # data_concat(analyzer, 'l2', l2_per_band)
            data_concat(analyzer, 'latent', latent)
            data_concat(analyzer, 'date', data_dict['date'])

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples
        for i, met in enumerate(metric_fns)
    })
    logger.info(log)

    # save the analyzer to csv using pandas
    columns = []
    for k in ['output', 'target', 'latent']:
        if k != 'latent':
            columns += [k+'_'+b for b in GPS]
        else:
            columns += [k+'_'+b for b in ATTRS]
    # TODO hstack the columns we want to save
    data = torch.hstack((
        analyzer['output'],
        analyzer['target'],
        # analyzer['l2'],
        analyzer['latent']
    ))
    if config['arch']['type'] in ['AE_Mogi_corr']:
        columns += ['init_output_'+b for b in GPS]
        columns += ['bias_'+b for b in GPS]
        data = torch.hstack((
            data,
            analyzer['init_output'],
            analyzer['bias']
        ))
    data = data.cpu().numpy()
    df = pd.DataFrame(columns=columns, data=data)
    df['date'] = analyzer['date']
    df.to_csv(str(config.resume).split('.pth')[0]+'_testset_analyzer.csv',
              index=False)
    logger.info('Analyzer saved to {}'.format(
        str(config.resume).split('.pth')[0]+'_testset_analyzer.csv'
    ))


def data_concat(analyzer: dict, key: str, data):
    if key not in analyzer:
        analyzer[key] = data
    elif type(data) == torch.Tensor:
        analyzer[key] = torch.cat((analyzer[key], data), dim=0)
    elif type(data) == list:
        analyzer[key] = analyzer[key] + data


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
