import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import pandas as pd
import numpy as np


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    # NOTE the test set needs to be set beforehand e.g. in dataset.py
    data_loader = getattr(module_data, config['data_loader']['type'])(
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
    loss_fn = getattr(module_loss, config['loss'])
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
    S2_BANDS = ['B02_BLUE', 'B03_GREEN', 'B04_RED', 'B05_RE1', 'B06_RE2',
                'B07_RE3', 'B08_NIR1', 'B8A_NIR2', 'B09_WV', 'B11_SWI1',
                'B12_SWI2']
    ATTRS = ['N', 'cab', 'cw', 'cm', 'LAI', 'LAIu', 'sd', 'fc']

    analyzer = {}

    with torch.no_grad():
        # for i, (data, target) in enumerate(tqdm(data_loader)):
        #     data, target = data.to(device), target.to(device)
        for batch_idx, data_dict in enumerate(data_loader):
            # TODO change the input and target keys
            data = data_dict[data_key].to(device)
            target = data_dict[target_key].to(device)
            output = model(data)

            #
            # save sample images, or do something with output here
            #

            # concatenate the loss per band to the loss_analyzer
            # calculate the squared loss of each element in the batch
            latent = model.encode(data)
            l2_per_band = torch.square(output-target)
            data_concat(analyzer, 'output', output)
            data_concat(analyzer, 'target', target)
            data_concat(analyzer, 'l2', l2_per_band)
            data_concat(analyzer, 'latent', latent)
            data_concat(analyzer, 'sample_id', data_dict['sample_id'])
            data_concat(analyzer, 'class', data_dict['class'])
            data_concat(analyzer, 'date', data_dict['date'])

            # if batch_idx == 0:
            #     analyzer['output'] = output
            #     analyzer['target'] = target
            #     analyzer['latent'] = latent
            #     analyzer['loss_per_band'] = l2_per_band
            #     analyzer['sample_id'] = data_dict['sample_id']
            #     analyzer['class'] = data_dict['class']
            #     analyzer['date'] = data_dict['date']
            # else:
            #     analyzer['loss_per_band'] = torch.cat((
            #         analyzer['loss_per_band'], l2_per_band
            #     ), dim=0)
            #     analyzer['sample_id'] = torch.cat((
            #         analyzer['sample_id'], data_dict['sample_id']
            #     ), dim=0)
            #     analyzer['class'] = analyzer['class'] + \
            #         data_dict['class']
            #     analyzer['date'] = analyzer['date'] + \
            #         data_dict['date']

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
    for k in ['output', 'target', 'l2', 'latent']:
        if k != 'latent':
            columns += [k+'_'+b for b in S2_BANDS]
        else:
            columns += [k+'_'+b for b in ATTRS]
    # TODO hstack the columns we want to save
    data = torch.hstack((
        analyzer['output'],
        analyzer['target'],
        analyzer['l2'],
        analyzer['latent']
    )).cpu().numpy()
    df = pd.DataFrame(columns=columns, data=data)
    df['sample_id'] = analyzer['sample_id']
    df['class'] = analyzer['class']
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
    # args.add_argument('-a', '--analyze', default=False, type=bool,
    #                   help='analyze and saved the test results (default: False)')

    config = ConfigParser.from_args(args)
    main(config)
