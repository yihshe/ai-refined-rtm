import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from rtm_torch.rtm import RTM


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


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
        assert hidden_dim == len(
            rtm_paras), "hidden_dim must be equal to the number of RTM parameters"
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
        # NOTE ["N", "cab", "cw", "cm", "LAI", "LAIu", "sd", "h", "cd"]
        self.rtm_paras = rtm_paras
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
        return self.encoder(x)

    #  define decode function to further process the output of decoder
    def decode(self, x):
        para_dict = {}
        for i, para_name in enumerate(self.rtm_paras.keys()):
            min = self.rtm_paras[para_name]['min']
            max = self.rtm_paras[para_name]['max']
            para_dict[para_name] = x[:, i]*(max-min)+min

        output = self.decoder.run(**para_dict)[:, self.bands_index]
        return (output-self.x_mean)/self.x_scale

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
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
            # nn.Sigmoid(),
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
