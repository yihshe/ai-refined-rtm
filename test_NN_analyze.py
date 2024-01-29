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
import json
from rtm_torch.rtm import RTM


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
    S2_FULL_BANDS = ['B01', 'B02_BLUE', 'B03_GREEN', 'B04_RED',
                    'B05_RE1', 'B06_RE2', 'B07_RE3', 'B08_NIR1',
                    'B8A_NIR2', 'B09_WV', 'B10', 'B11_SWI1',
                    'B12_SWI2']
    ATTRS = ['N', 'cab', 'cw', 'cm', 'LAI', 'LAIu', 'fc']

    # load rtm_paras
    rtm_paras = json.load(open("/maps/ys611/ai-refined-rtm/configs/rtm_paras.json"))
    assert ATTRS == list(rtm_paras.keys()), "ATTRS must be consistent with rtm_paras"
    rtm = RTM()
    
    x_mean = torch.tensor(
        np.load("/maps/ys611/ai-refined-rtm/data/synthetic/20240124/train_x_mean.npy")
        ).float().unsqueeze(0).to(device)
    x_scale = torch.tensor(
        np.load("/maps/ys611/ai-refined-rtm/data/synthetic/20240124/train_x_scale.npy")
        ).float().unsqueeze(0).to(device)
    bands_index = [i for i in range(
        len(S2_FULL_BANDS)) if S2_FULL_BANDS[i] not in ['B01', 'B10']]
    S2_BANDS = [S2_FULL_BANDS[i] for i in bands_index]
    
    analyzer = {}

    if config['data_loader']['type'] == 'SyntheticS2DataLoader':
        mode = "test"
    elif config['data_loader']['type'] == 'SpectrumS2DataLoader':
        mode = "infer"

    if mode == "test":
        """
        For test mode, NNRegressor will be evaluated on the synthetic test set,
        where the predicted variables will be compared with the sampled variables
        and MSE loss will be calculated.
        """
        with torch.no_grad():
            for batch_idx, data_dict in enumerate(data_loader):
                data = data_dict[data_key].to(device)
                target = data_dict[target_key].to(device)
                output = model(data)
                # concatenate the loss per band to the loss_analyzer
                l2_per_band = torch.square(output-target)
                data_concat(analyzer, 'output', output)
                data_concat(analyzer, 'target', target)
                data_concat(analyzer, 'l2', l2_per_band)

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
        for k in ['output', 'target', 'l2']:
            columns += [k+'_'+b for b in ATTRS]

        # NOTE all the output and target variales here are saved in the normalized scale ranging from 0 to 1
        data = torch.hstack((
            analyzer['output'],
            analyzer['target'],
            analyzer['l2'],
        )).cpu().numpy()
        df = pd.DataFrame(columns=columns, data=data)
        df.to_csv(str(config.resume).split('.pth')[0]+'_testset_analyzer_syn.csv',
                  index=False)
        logger.info('Analyzer saved to {}'.format(
            str(config.resume).split('.pth')[0]+'_testset_analyzer_syn.csv'
        ))

    elif mode == "infer":
        """
        For infer mode, NNRegressor will be evaluated on the real test set,
        where the predicted variables will be saved, and the reconstructed spectra
        will be compared with the real spectra and the corresponding MSE loss will
        be calculated.
        Note that spectra of the real test set has been standardized using the
        mean and std of the synthetic training set, on which the NNRegressor is
        trained. 
        """
        # Reconstruction loss will also be calculated in the inference mode
        with torch.no_grad():
            for batch_idx, data_dict in enumerate(data_loader):
                data = data_dict[data_key].to(device)
                # target = data_dict[target_key].to(device)
                latent = model(data)
                # concatenate the loss per band to the loss_analyzer
                data_concat(analyzer, 'latent', latent)
                data_concat(analyzer, 'sample_id', data_dict['sample_id'])
                data_concat(analyzer, 'class', data_dict['class'])
                data_concat(analyzer, 'date', data_dict['date'])

                # computing the reconstructed spectra
                spectra = vars2spectra(latent, rtm, rtm_paras, x_mean, x_scale, bands_index)
                data_concat(analyzer, 'output', spectra)
                data_concat(analyzer, 'target', data)
                
                # computing the reconstruction loss
                loss = loss_fn(spectra, data)
                batch_size = data.shape[0]
                total_loss += loss.item() * batch_size
                for i, metric in enumerate(metric_fns):
                    total_metrics[i] += metric(spectra, data) * batch_size
        n_samples = len(data_loader.sampler)
        log = {'reconstruction loss': total_loss / n_samples}
        log.update({
            met.__name__: total_metrics[i].item() / n_samples
            for i, met in enumerate(metric_fns)
        })
        logger.info(log)

        # save the analyzer to csv using pandas
        columns = []
        for k in ['output', 'target', 'latent']:
            if k != 'latent':
                columns += [k+'_'+b for b in S2_BANDS]
            else:
                columns += [k+'_'+b for b in ATTRS]
        data = torch.hstack((
            analyzer['output'],
            analyzer['target'],
            analyzer['latent']
        )).cpu().numpy()
        df = pd.DataFrame(columns=columns, data=data)
        # add the sample_id, class, and date from the real testset
        df['sample_id'] = analyzer['sample_id']
        df['class'] = analyzer['class']
        df['date'] = analyzer['date']
        df.to_csv(str(config.resume).split('.pth')[0]+'_testset_analyzer_real.csv',
                  index=False)
        logger.info('Analyzer saved to {}'.format(
            str(config.resume).split('.pth')[0]+'_testset_analyzer_real.csv'
        ))


def data_concat(analyzer: dict, key: str, data):
    if key not in analyzer:
        analyzer[key] = data
    elif type(data) == torch.Tensor:
        analyzer[key] = torch.cat((analyzer[key], data), dim=0)
    elif type(data) == list:
        analyzer[key] = analyzer[key] + data

def vars2spectra(x: torch.tensor,rtm: RTM, rtm_paras: dict, 
                 x_mean: torch.tensor, x_scale: torch.tensor, 
                 bands_index: list):
    # convert the variables to spectra
    # x: [batch_size, n_vars]
    # rtm_paras: dict
    # return: [batch_size, n_bands]
    para_dict = {}
    for i, para_name in enumerate(rtm_paras.keys()):
        min = rtm_paras[para_name]['min']
        max = rtm_paras[para_name]['max']
        para_dict[para_name] = x[:, i]*(max-min)+min
    assert 'fc' in para_dict.keys(), "fc must be included in the rtm_paras"
    # calculate cd from sd and fc
    SD = 500
    para_dict['cd'] = torch.sqrt(
        (para_dict['fc']*10000)/(torch.pi*SD))*2
    para_dict['h'] = torch.exp(
        2.117 + 0.507*torch.log(para_dict['cd']))
    # run the rtm
    spectra = rtm.run(**para_dict)[:, bands_index]
    # standardize the spectra
    return (spectra-x_mean)/x_scale



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
