import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from base import BaseModel
from rtm_torch.rtm import RTM

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

class EmbeddingModule(nn.Module):
    def __init__(self, n_species, species_embedding_dim):
        super(EmbeddingModule, self).__init__()
        self.species_embedding = nn.Embedding(n_species, species_embedding_dim)

    def forward(self, species_idx, group_idx, sin_date, cos_date):
        species_embed = self.species_embedding(species_idx)
        
        # Assuming sin_date and cos_date are already tensors and have the correct shape
        # Concatenate all embeddings and cyclical date features
        combined_embed = torch.cat([species_embed, group_idx.unsqueeze(-1), 
                                    sin_date.unsqueeze(-1), 
                                    cos_date.unsqueeze(-1)], dim=1).float()
        return combined_embed

class AE_RTM_con(BaseModel):
    """
    Vanilla AutoEncoder (AE) with RTM as the decoder
    input -> encoder (learnable) -> decoder (INFORM) -> output
    Conditional AutoEncoder (CAE) with RTM as the decoder
    """

    def __init__(self, input_dim, hidden_dim, rtm_paras, standardization,
                 n_species, species_embedding_dim):
        super().__init__()
        input_dim += species_embedding_dim + 3
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
        
        # Embedding module
        self.embedding_module = EmbeddingModule(n_species, species_embedding_dim)

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

    def embed(self, species_idx, group_idx, sin_date, cos_date):
        embedded = self.embedding_module(species_idx, group_idx, sin_date, cos_date)
        return embedded
    
    def forward(self, x, species_idx, group_idx, sin_date, cos_date):
        # Embedding
        embedded = self.embed(species_idx, group_idx, sin_date, cos_date)
        para_dict = self.encode(torch.cat([x, embedded], dim=1))
        x = self.decode(para_dict)
        return x
    
    def infer(self, x, species_idx, group_idx, sin_date, cos_date):
        embedded = self.embed(species_idx, group_idx, sin_date, cos_date)
        x = self.encode(torch.cat([x, embedded], dim=1))
        return x

class AE_RTM_corr_con(AE_RTM_con):
    """
    Vanilla AutoEncoder (AE) with RTM as the decoder and additional layers for correction
    input -> encoder (learnable) -> decoder (INFORM) -> correction -> output
    Conditional AutoEncoder (CAE) with RTM as the decoder and additional layers for correction
    """

    def __init__(self, input_dim, hidden_dim, rtm_paras, standardization,
                 n_species, species_embedding_dim):
        super().__init__(input_dim, hidden_dim, rtm_paras, standardization,
                         n_species, species_embedding_dim)
        self.correction = nn.Sequential(
            nn.Linear(len(self.bands_index)+species_embedding_dim+3, 
                      4*len(self.bands_index)),
            nn.ReLU(),
            nn.Linear(4*len(self.bands_index), len(self.bands_index)),
        )

    def correct(self, x):
        return self.correction(x)

    def forward(self, x, species_idx, group_idx, sin_date, cos_date):
        embedded = self.embed(species_idx, group_idx, sin_date, cos_date)
        x = self.encode(torch.cat([x, embedded], dim=1))
        x = self.decode(x)
        x = self.correct(torch.cat([x, embedded], dim=1))
        return x