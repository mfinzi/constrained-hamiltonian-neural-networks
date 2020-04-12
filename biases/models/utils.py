import torch
import torch.nn as nn


def FCtanh(chin, chout):
    return nn.Sequential(nn.Linear(chin, chout), nn.Tanh())


def FCswish(chin, chout):
    return nn.Sequential(nn.Linear(chin, chout), nn.Swish())


def FCsoftplus(chin, chout):
    return nn.Sequential(nn.Linear(chin, chout), nn.Softplus())


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


def tril_mask(square_mat):
    n = square_mat.size(-1)
    coords = square_mat.new(n)
    torch.arange(n, out=coords)
    return coords <= coords.view(n, 1)
