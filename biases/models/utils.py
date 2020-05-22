import torch
import torch.nn as nn
from lie_conv.lieConv import Swish
import math


def FCtanh(chin, chout, zero_bias=False, orthogonal_init=False):
    return nn.Sequential(Linear(chin, chout, zero_bias, orthogonal_init), nn.Tanh())


def FCswish(chin, chout, zero_bias=False, orthogonal_init=False):
    return nn.Sequential(Linear(chin, chout, zero_bias, orthogonal_init), Swish())


def FCsoftplus(chin, chout, zero_bias=False, orthogonal_init=False):
    return nn.Sequential(Linear(chin, chout, zero_bias, orthogonal_init), nn.Softplus())


def Linear(chin, chout, zero_bias=False, orthogonal_init=False):
    linear = nn.Linear(chin, chout)
    if zero_bias:
        torch.nn.init.zeros_(linear.bias)
    if orthogonal_init:
        torch.nn.init.orthogonal_(linear.weight)
    return linear


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class CosSin(nn.Module):
    def __init__(self, q_ndim, angular_dims, only_q=True):
        super(CosSin, self).__init__()
        self.q_ndim = q_ndim
        self.angular_dims = tuple(angular_dims)
        self.non_angular_dims = tuple(set(range(q_ndim)) - set(angular_dims))
        self.only_q = only_q

    def forward(self, q_or_qother):
        if self.only_q:
            q = q_or_qother
        else:
            q, other = q_or_qother.chunk(2, dim=-1)
        assert q.size(-1) == self.q_ndim


        q_angular = q[..., self.angular_dims]
        q_not_angular = q[..., self.non_angular_dims]

        cos_ang_q, sin_ang_q = torch.cos(q_angular), torch.sin(q_angular)
        q = torch.cat([cos_ang_q, sin_ang_q, q_not_angular], dim=-1)

        if self.only_q:
            q_or_other = q
        else:
            q_or_other = torch.cat([q, other], dim=-1)

        return q_or_other


def tril_mask(square_mat):
    n = square_mat.size(-1)
    coords = torch.arange(n)
    return coords <= coords.view(n, 1)


def mod_angles(q, angular_dims):
    assert q.ndim == 2
    D = q.size(-1)
    non_angular_dims = list(set(range(D)) - set(angular_dims))
    # Map to -pi, pi
    q_modded_dims = torch.fmod(q[..., angular_dims] + math.pi, 2 * math.pi) + (2. * (q[..., angular_dims] < -math.pi) - 1) * math.pi
    if (q_modded_dims.abs() > math.pi).any():
        raise RuntimeError("Angles beyond [-pi, pi]!")
    q_non_modded_dims = q[..., non_angular_dims]
    return torch.cat([q_modded_dims, q_non_modded_dims], dim=-1)
