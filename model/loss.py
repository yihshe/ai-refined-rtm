import torch
import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)


def mse_loss(output, target):
    return F.mse_loss(output, target)


def mse_loss_per_band(output, target):
    with torch.no_grad():
        return torch.mean(torch.square(output - target), dim=0)
