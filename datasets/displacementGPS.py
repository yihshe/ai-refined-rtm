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

        return data_dict
    