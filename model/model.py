import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


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

# build a vanilla AutoEncoder based on BaseModel with the following architecture in PyTorch:
# input -> encoder -> decoder -> output
# input and output are the same 1-D vector of size input_dim less than 100
# encoder is 3 fully connected layers with ReLU activation with suitable hidden dimensions as you see fit
# the output of encoder is the latent vector of size hidden_dim
# decoder is 3 fully connected layers with ReLU activation
# the output of decoder is the reconstructed vector of size input_dim
# the loss function is the mean squared error between input and output
# the optimizer is Adam with learning rate 1e-3
# the number of epochs is 100
# the batch size is 128
# build a class VanillaAE(BaseModel) in model.py


class VanillaAE(BaseModel):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            # nn.Linear(input_dim, 64),
            # nn.ReLU(),
            # nn.Linear(64, 32),
            # nn.ReLU(),
            # nn.Linear(32, hidden_dim),
            # nn.ReLU(),
            # NOTE the above is the original encoder
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim),
            nn.ReLU(),
        )
        # TODO replace the encoder with the fixed RTM from rtm.py
        self.decoder = nn.Sequential(
            # nn.Linear(hidden_dim, 32),
            # nn.ReLU(),
            # nn.Linear(32, 64),
            # nn.ReLU(),
            # nn.Linear(64, input_dim),
            # nn.ReLU(),
            # NOTE the above is the original decoder
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            # nn.ReLU(),
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

    # def loss_function(self, x, x_hat):
    #     # compute the mean squared error between input and output
    #     return F.mse_loss(x_hat, x)
