import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from base import BaseModel
from rtm_torch.rtm import RTM
from mogi.mogi import Mogi
from utils.util import MemoryBank


class VanillaAE(BaseModel):
    """
    Vanilla AutoEncoder (AE) 
    input -> encoder -> decoder -> output
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, hidden_dim),
            nn.ReLU(),
        )
        # TODO modify hidden_dim to 10, add ReLU to decoder and run it again
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    #  define encode function to further process the output of encoder
    def encode(self, x):
        # TODO add a linear layer to map the output of encoder to the latent biophysical variables
        # add a sigmoid function to map the output of the linear layer to the range [0,1]
        return self.encoder(x)

    #  define decode function to further process the output of decoder
    def decode(self, x):
        # TODO add a linear layer to map the output of the rtm
        # Do we need ReLU at the final layer?
        return self.decoder(x)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


class AE_RTM(BaseModel):
    """
    Vanilla AutoEncoder (AE) with RTM as the decoder
    input -> encoder (learnable) -> decoder (INFORM) -> output
    """

    def __init__(self, input_dim, hidden_dim, rtm_paras, standardization):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # The encoder is learnable neural networks
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, hidden_dim),
            nn.Sigmoid(),
        )
        # The decoder is the INFORM RTM with fixed parameters
        self.decoder = RTM()
        # NOTE output of rtm_paras from the encoder:
        # ["N", "cab", "cw", "cm", "LAI", "LAIu", "sd", "h", "fc"]
        # then, cd will be calculated from sd and fc
        self.rtm_paras = json.load(open(rtm_paras))
        assert hidden_dim == len(
            self.rtm_paras), "hidden_dim must be equal to the number of RTM parameters"
        S2_FULL_BANDS = ['B01', 'B02_BLUE', 'B03_GREEN', 'B04_RED',
                         'B05_RE1', 'B06_RE2', 'B07_RE3', 'B08_NIR1',
                         'B8A_NIR2', 'B09_WV', 'B10', 'B11_SWI1',
                         'B12_SWI2']
        self.bands_index = [i for i in range(
            len(S2_FULL_BANDS)) if S2_FULL_BANDS[i] not in ['B01', 'B10']]
        # Mean and scale for standardization
        self.device = self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.x_mean = torch.tensor(
            np.load(standardization['x_mean'])).float().unsqueeze(0).to(self.device)
        self.x_scale = torch.tensor(
            np.load(standardization['x_scale'])).float().unsqueeze(0).to(self.device)

    #  define encode function to further process the output of encoder
    def encode(self, x):
        x = self.encoder(x)
        para_dict = {}
        for i, para_name in enumerate(self.rtm_paras.keys()):
            min = self.rtm_paras[para_name]['min']
            max = self.rtm_paras[para_name]['max']
            para_dict[para_name] = x[:, i]*(max-min)+min
        assert 'fc' in para_dict.keys(), "fc must be included in the rtm_paras"
        # calculate cd from sd and fc
        SD = 500
        para_dict['cd'] = torch.sqrt(
            (para_dict['fc']*10000)/(torch.pi*SD))*2
        para_dict['h'] = torch.exp(
            2.117 + 0.507*torch.log(para_dict['cd']))
        return para_dict
        # return self.encoder(x)

    #  define decode function to further process the output of decoder
    def decode(self, para_dict):
        output = self.decoder.run(**para_dict)[:, self.bands_index]
        return (output-self.x_mean)/self.x_scale

    def forward(self, x):
        para_dict = self.encode(x)
        x = self.decode(para_dict)
        return x


class AE_RTM_corr(AE_RTM):
    """
    Vanilla AutoEncoder (AE) with RTM as the decoder and additional layers for correction
    input -> encoder (learnable) -> decoder (INFORM) -> correction -> output
    """

    def __init__(self, input_dim, hidden_dim, rtm_paras, standardization):
        super().__init__(input_dim, hidden_dim, rtm_paras, standardization)
        self.correction = nn.Sequential(
            nn.Linear(len(self.bands_index), 4*len(self.bands_index)),
            nn.ReLU(),
            nn.Linear(4*len(self.bands_index), len(self.bands_index)),
        )

    def correct(self, x):
        return self.correction(x)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        x = self.correct(x)
        return x


class NNRegressor(BaseModel):
    """
    Approximate Neural Network (ANN) with PyTorch
    input -> encoder -> decoder -> output
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, hidden_dim),
            nn.Sigmoid(),
        )

    #  define encode function to further process the output of encoder
    def encode(self, x):
        # output of the encoder is just an NNRegressor
        # NOTE learning of latents should be conducted in normalized space
        return self.encoder(x)

    def forward(self, x):
        # TODO forward mode can be either "train" or "infer" in the future
        # so far it is only "train" thus the output is a scale factor
        x = self.encode(x)
        return x


class AE_Mogi(BaseModel):
    """
    Vanilla AutoEncoder (AE) with Mogi as the decoder
    input -> encoder (learnable) -> decoder (INFORM) -> output
    """

    def __init__(self, input_dim, hidden_dim, mogi_paras, station_info,
                 standardization):
        super().__init__()
        self.input_dim = input_dim  # 36
        self.hidden_dim = hidden_dim  # 5
        # The encoder is learnable neural networks
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, hidden_dim),
            nn.Sigmoid(),
        )
        #
        # The decoder is the INFORM RTM with fixed parameters
        self.station_info = json.load(open(station_info))
        x = torch.tensor([self.station_info[k]['xE']
                         for k in self.station_info.keys()])*1000  # m
        y = torch.tensor([self.station_info[k]['yN']
                         for k in self.station_info.keys()])*1000  # m
        self.decoder = Mogi(x, y)

        self.mogi_paras = json.load(open(mogi_paras))
        assert hidden_dim == len(
            self.mogi_paras), "hidden_dim must be equal to the number of Mogi parameters"
        # Mean and scale for standardization of model input
        self.device = self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.x_mean = torch.tensor(
            np.load(standardization['x_mean'])).float().unsqueeze(0).to(self.device)
        self.x_scale = torch.tensor(
            np.load(standardization['x_scale'])).float().unsqueeze(0).to(self.device)

    #  define encode function to further process the output of encoder
    def encode(self, x):
        # x = torch.cat((x, sin, cos), dim=1)
        x = self.encoder(x)
        return x

    def transform(self, x):
        """
        Transform the output of encoder to the physical parameters
        """
        para_dict = {}
        for i, para_name in enumerate(self.mogi_paras.keys()):
            min = self.mogi_paras[para_name]['min']
            max = self.mogi_paras[para_name]['max']
            # if x shape is (batch, sequence, feature), then x[:,:,i] is the i-th feature
            if len(x.shape) == 3:
                para_dict[para_name] = x[:, :, i]*(max-min)+min
            else:
                para_dict[para_name] = x[:, i]*(max-min)+min
            if para_name in ['xcen', 'ycen', 'd']:
                para_dict[para_name] = para_dict[para_name]*1000

        # para_dict['dV'] = para_dict['dV_factor'] * \
        #     torch.pow(10, para_dict['dV_power'])  # m^3
        para_dict['dV'] = para_dict['dV'] * \
            torch.pow(10, torch.tensor(5)) - torch.pow(10, torch.tensor(7))

        return para_dict

    #  define decode function to further process the output of decoder
    def decode(self, para_dict):
        # if within a batch, the input is a sequence, then iterate over sequence
        # and the output shape should be (batch, sequence, feature)
        if len(para_dict['xcen'].shape) == 2:
            output = torch.stack(
                [self.decoder.run(
                    para_dict['xcen'][i], para_dict['ycen'][i],
                    para_dict['d'][i], para_dict['dV'][i]
                ) for i in range(para_dict['xcen'].shape[0])]
            )
        else:
            # output in mm, same as the input
            output = self.decoder.run(para_dict['xcen'], para_dict['ycen'],
                                      para_dict['d'], para_dict['dV'])
        # NOTE scaling the output with mean and scale helps the model to learn better
        return (output-self.x_mean)/self.x_scale

    # def forward(self, x):
    #     para_dict = self.encode(x)
    #     x = self.decode(para_dict)
    #     return x

    def forward(self, x):
        x0 = self.encode(x)
        x1 = self.transform(x0)
        x2 = self.decode(x1)
        return x0, x1, x2


class AE_Mogi_corr(AE_Mogi):
    """
    Vanilla AutoEncoder (AE) with Mogi as the decoder and additional layers for correction
    input -> encoder (learnable) -> decoder (Mogi) -> correction -> output
    """

    def __init__(self, input_dim, hidden_dim, mogi_paras, station_info,
                 standardization):
        super().__init__(input_dim, hidden_dim, mogi_paras, station_info,
                         standardization)
        self.correction = nn.Sequential(
            nn.Linear(input_dim, 4*input_dim),
            nn.ReLU(),
            nn.Linear(4*input_dim, input_dim),
        )

    def correct(self, x):
        # x = torch.cat((x, sin, cos), dim=1)
        return self.correction(x)

    # def forward(self, x):
    #     x = self.encode(x)
    #     x = self.decode(x)
    #     x = self.correct(x)
    #     return x
    def forward(self, x):
        x0 = self.encode(x)
        x1 = self.transform(x0)
        x2 = self.decode(x1)
        x3 = self.correct(x2)
        return x0, x1, x2, x3
        # return x3
