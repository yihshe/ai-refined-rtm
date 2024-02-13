import pandas as pd
import torch
import torch.utils.data as data
import numpy as np


class SpectrumS2(data.Dataset):
    def __init__(self, csv_path, transform=None):
        super(SpectrumS2, self).__init__()
        # the dataset is a tabular data of satellite spectrums from Sentinel2
        self.data_df = pd.read_csv(csv_path)
        self.transform = transform
        # TODO make sure the bands are in the same order as the RTM output
        self.s2_bands = ['B02_BLUE', 'B03_GREEN', 'B04_RED', 'B05_RE1',
                         'B06_RE2', 'B07_RE3', 'B08_NIR1', 'B8A_NIR2',
                         'B09_WV', 'B11_SWI1', 'B12_SWI2']
        self.attr_info = ['class', 'sample_id', 'date']
        # self.attr_info = ['class', 'sample_id', 'date', 
        #                   'sin_date', 'cos_date', 'species_idx', 'group_idx']
        # NOTE currently the ground truth of latent variables are not available

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        sample = self.data_df.iloc[index]
        spectrum = torch.tensor(
            sample[self.s2_bands].values.astype('float32')
        ).to(torch.float32)

        if self.transform is not None:
            spectrum = self.transform(spectrum)
        attrs = sample[self.attr_info].values

        data_dict = {'spectrum': spectrum}
        data_dict.update({k: v for k, v in zip(self.attr_info, attrs)})

        return data_dict


class SyntheticS2(data.Dataset):
    def __init__(self, csv_path, transform=None):
        super(SyntheticS2, self).__init__()
        # the dataset is a tabular data of satellite spectrums from Sentinel2
        self.data_df = pd.read_csv(csv_path)
        self.transform = transform
        # TODO make sure the bands are in the same order as the RTM output
        self.s2_bands = ['B02_BLUE', 'B03_GREEN', 'B04_RED', 'B05_RE1',
                         'B06_RE2', 'B07_RE3', 'B08_NIR1', 'B8A_NIR2',
                         'B09_WV', 'B11_SWI1', 'B12_SWI2']
        self.rtm_paras = ['N', 'cab', 'cw', 'cm', 'LAI', 'LAIu', 'fc']
    
    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        sample = self.data_df.iloc[index]
        spectrum = torch.tensor(
            sample[self.s2_bands].values.astype('float32')
        ).to(torch.float32)

        if self.transform is not None:
            spectrum = self.transform(spectrum)
        rtm_paras = sample[self.rtm_paras].values.astype('float32')

        data_dict = {
            'spectrum': spectrum,
            'rtm_paras': rtm_paras
        }

        return data_dict
