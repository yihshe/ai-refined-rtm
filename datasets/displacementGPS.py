import numpy as np
import pandas as pd
import torch
import torch.utils.data as data


class DisplacementGPS(data.Dataset):
    def __init__(self, csv_path):
        super(DisplacementGPS, self).__init__()
        # the dataset is a tabular data of GPS displacement data
        # each row is displacements from 12 stations at the same time point
        self.data_df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        sample = self.data_df.iloc[index]
        data_dict = {}
        data_dict['displacement'] = torch.tensor(
            sample[:36].values.astype('float32')
        ).to(torch.float32)
        data_dict['date'] = sample[-3]
        for k in ['sin_date', 'cos_date']:
            data_dict[k] = torch.tensor(
                sample[k].astype('float32')
            ).to(torch.float32).unsqueeze(0)

        return data_dict

# dataset by slicing data into sequences
class DisplacementGPSSeq(data.Dataset):
    def __init__(self, csv_path, seq_len=5, step_size=5):
        super(DisplacementGPSSeq, self).__init__()
        self.seq_len = seq_len
        self.step_size = step_size
        self.data_df = pd.read_csv(csv_path)
        self.data_df['date'] = pd.to_datetime(self.data_df['date'])
        self.data_df = self.data_df.sort_values(by='date')
        # self.data_df['date'] = self.data_df['date'].dt.strftime('%Y-%m-%d')

    def __len__(self):
        # return len(self.data_df) - self.seq_len + 1
        return (len(self.data_df) - self.seq_len) // self.step_size + 1

    def __getitem__(self, idx):
        index = idx * self.step_size
        sample = self.data_df.iloc[index:index+self.seq_len]
        data_dict = {}
        data_dict['displacement'] = torch.tensor(
            sample.iloc[:, :36].values.astype('float32')
        ).to(torch.float32)
        # NOTE this dataloader is used for training, without loading date
        # data_dict['date'] = sample['date'].values
        for k in ['sin_date', 'cos_date']:
            data_dict[k] = torch.tensor(
                sample[k].values.astype('float32')
            ).to(torch.float32)

        return data_dict
