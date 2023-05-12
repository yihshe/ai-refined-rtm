import torch.nn as nn
import numpy as np
from abc import abstractmethod
from .types_ import *


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    def __init__(self):
        super(BaseModel, self).__init__()

    def encode(self, *inputs):
        """
        Encodes input into latent space
        """
        raise NotImplementedError
    
    def decode(self, *inputs):
        """
        Decodes latent space into output
        """
        raise NotImplementedError
    
    def sample(self, *inputs):
        """
        Samples from latent space
        """
        raise NotImplementedError
    
    def generate(self, *inputs):
        """
        Generates output from samples in latent space
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError
    
    @abstractmethod
    def loss_function(self, *inputs, **kwargs):
        """
        Computes the loss given inputs

        :return: Loss tensor
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
