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


def mse_loss_mogi_reg(output, target, memory_bank: MemoryBank = None, alpha=1e-2):
    corrected_output = output[-1]
    mogi_output = output[-2]
    mse_loss = F.mse_loss(corrected_output, target)
    # reg_loss = lambda_reg * F.mse_loss(mogi_output, torch.zeros_like(mogi_output))
    # return mse_loss + reg_loss
    smoothness_reg = smoothness_loss(output[-2])
    if memory_bank is None:
        return mse_loss
    else:
        latents = output[0]
        # regularize the predictions of xcen, ycen, and d so that they are roughly constant through out different batches and sequences
        xcen = latents[:, :, 0]
        ycen = latents[:, :, 1]
        d = latents[:, :, 2]

        # # calculate the variance for the mogi location across batch and sequences
        # xcen_var = torch.var(xcen)
        # ycen_var = torch.var(ycen)
        # d_var = torch.var(d)
        # consistency_reg = xcen_var + ycen_var + d_var

        # calculate the current batch mean and variance for the mogi location
        xcen_mean = torch.mean(xcen)
        ycen_mean = torch.mean(ycen)
        d_mean = torch.mean(d)

        current_batch_mean = torch.tensor([xcen_mean, ycen_mean, d_mean])
        memory_bank.update(current_batch_mean)

        variance_loss = torch.mean(torch.std(
            memory_bank.get() - current_batch_mean))

        # total_loss = mse_loss + alpha * smoothness_reg + alpha * variance_loss
        total_loss = mse_loss + alpha * variance_loss
        return total_loss


def smoothness_loss(output):
    # calculate the difference between adjacent elements of a sequence
    # the output has a shape of (batch, sequence, channels)
    diff = output[:, 1:, :] - output[:, :-1, :]
    return torch.mean(torch.square(diff))
