import torch
import torch.nn as nn
from torchdiffeq import odeint
from lie_conv.utils import export, Named
from biases.models.utils import FCswish
import numpy as np


@export
class NN(nn.Module, metaclass=Named):
    def __init__(self, G, d=1, k=300, num_layers=4, angular_dims=[], **kwargs):
        super().__init__()
        n = len(G.nodes())
        chs = [n * 2 * d] + num_layers * [k]
        self.net = nn.Sequential(
            *[FCswish(chs[i], chs[i + 1]) for i in range(num_layers)],
            nn.Linear(chs[-1], 2 * d * n)
        )
        self.nfe = 0
        self.angular_dims = list(range(n * d)) if angular_dims == True else angular_dims

    def forward(self, t, z, wgrad=True):
        D = z.shape[-1] // 2
        theta_mod = (z[..., self.angular_dims] + np.pi) % (2 * np.pi) - np.pi
        not_angular_dims = list(set(range(D)) - set(self.angular_dims))
        not_angular_q = z[..., not_angular_dims]
        z_mod = torch.cat([theta_mod, not_angular_q, z[..., D:]], dim=-1)
        return self.net(z_mod)

    def integrate(self, z0, ts, tol=1e-4):
        """ inputs [z0: (bs, z_dim), ts: (bs, T), sys_params: (bs, n, c)]
            outputs pred_zs: (bs, T, z_dim) """
        bs = z0.shape[0]
        zt = odeint(self, z0.reshape(bs, -1), ts, rtol=tol, method="rk4").permute(
            1, 0, 2
        )
        return zt.reshape(bs, len(ts), *z0.shape[1:])


@export
class DeltaNN(NN):
    def integrate(self, z0, ts, tol=1e-4):
        """ inputs [z0: (bs, z_dim), ts: (bs, T), sys_params: (bs, n, c)]
            outputs pred_zs: (bs, T, z_dim) """
        bs = z0.shape[0]
        dts = ts[1:] - ts[:-1]
        zts = [z0.reshape(bs, -1)]
        for dt in dts:
            zts.append(zts[-1] + dt * self(None, zts[-1]))
        return torch.stack(zts, dim=1).reshape(bs, len(ts), *z0.shape[1:])
