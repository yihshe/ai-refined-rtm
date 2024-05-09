import torch
import torch.nn.functional as F
from utils.util import MemoryBank


def nll_loss(output, target):
    return F.nll_loss(output, target)


def mse_loss(output, target):
    return F.mse_loss(output, target)


def mse_loss_per_channel(output, target):
    with torch.no_grad():
        return torch.mean(torch.square(output - target), dim=0)

def mse_loss_mogi(output, target):
    final_output = output[-1]
    return F.mse_loss(final_output, target)

def mse_loss_mogi_reg(output, target, alpha=1e-2):
    final_output = output[-1]
    mogi_output = output[2]
    mse_loss = F.mse_loss(final_output, target)

    smoothness_reg = smoothness_loss(mogi_output)

    total_loss = mse_loss + alpha * smoothness_reg 
    return total_loss


def smoothness_loss(output):
    # calculate the difference between adjacent elements of a sequence
    # the output has a shape of (batch, sequence, channels)
    diff = output[:, 1:, :] - output[:, :-1, :]
    return torch.mean(torch.square(diff))
