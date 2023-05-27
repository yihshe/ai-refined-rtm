from torch.autograd import gradcheck
import torch
from torch.autograd import Function
from scipy.special import exp1 as scipy_exp1

# Custom autograd function for exponential integral function E1


class Exp1(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.tensor(scipy_exp1(input.detach().numpy()), dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        epsilon = 1e-7  # a small constant
        grad_input = grad_output * (-torch.exp(-input) / (input + epsilon))
        return grad_input


def exp1(input):
    return Exp1.apply(input)


# # autograd check for exp1
# input = torch.rand(20, 20, dtype=torch.double, requires_grad=True)
# test = gradcheck(exp1, (input,), eps=1e-6, atol=1e-4)
# print(test)
