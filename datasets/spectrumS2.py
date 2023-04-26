import pandas as pd
import torch
import torch.utils.data as data

# TODO split the dataset into train and test sets.
# The dataloaders for each set can be set up at the config file (with respective data path)
# The validation set can be set up during the training process when creating the trainer object


class SpectrumS2(data.Dataset):
    def __init__(self, csv_path, transform=None):
        super(SpectrumS2, self).__init__()
        # the dataset is a tabular data of satellite spectrums from Sentinel2
        self.data_df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        # TODO make sure order of the bands are same as the output of the RTM
        spectrum = torch.tensor(
            self.data_df.iloc[index].values.astype('float32')
        )
        if self.transform is not None:
            spectrum = self.transform(spectrum)
        # TODO return the specturms and any latent variables if needed
        return spectrum
