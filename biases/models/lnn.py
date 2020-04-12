import torch
import torch.nn as nn
from torchdiffeq import odeint
from lie_conv.utils import export, Named
from biases.models.utils import FCswish, Reshape
from biases.dynamics import LagrangianDynamics
import numpy as np


@export
class LNN(nn.Module, metaclass=Named):
    def __init__(self, G, hidden_size=256, num_layers=4, angular_dims=[], **kwargs):
        super().__init__(**kwargs)
        # Number of function evaluations
        self.nfe = 0
        # Number of degrees of freedom
        self.n = n = len(G.nodes)
        chs = [2 * n] + num_layers * [hidden_size]
        print("LNN currently ignores time as an input")
        self.net = nn.Sequential(
            *[FCswish(chs[i], chs[i + 1]) for i in range(num_layers)],
            nn.Linear(chs[-1], 1),
            Reshape(-1)
        )
        self.angular_dims = list(range(n)) if angular_dims == True else angular_dims

    def forward(self, t, z, wgrad=True):
        """ inputs: [t (T,)], [z (bs,2n)]. Outputs: [F (bs,2n)]"""
        self.nfe += 1
        dynamics = LagrangianDynamics(self.L, wgrad=wgrad)
        # print(t.shape)
        # print(z.shape)
        return dynamics(t, z)

    def L(self, t, z):
        """ inputs: [t (T,)], [z (bs,2nd)]. Outputs: [H (bs,)]"""
        D = z.shape[-1] // 2
        theta_mod = (z[..., self.angular_dims] + np.pi) % (2 * np.pi) - np.pi
        not_angular_dims = list(set(range(D)) - set(self.angular_dims))
        not_angular_q = z[..., not_angular_dims]
        p = z[..., D:]
        z_mod = torch.cat([theta_mod, not_angular_q, p], dim=-1)
        return self.net(z_mod) + 1e-1 * (p * p).sum(-1)

    def integrate(self, z0, ts, tol=1e-4):
        """ inputs: [z0 (bs,2,n,d)], [ts (T,)]. Outputs: [xvt (bs,T,2,n,d)]"""
        bs = z0.shape[0]
        xvt = odeint(self, z0.reshape(bs, -1), ts, rtol=tol, method="rk4").permute(
            1, 0, 2
        )
        return xvt.reshape(bs, len(ts), *z0.shape[1:])
