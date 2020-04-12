import torch
import torch.nn as nn
from torchdiffeq import odeint
from lie_conv.utils import export, Named
from biases.models.utils import FCswish
import numpy as np
from typing import Tuple, Union


@export
class NN(nn.Module, metaclass=Named):
    def __init__(
        self,
        G,
        dof_ndim: int = 1,
        hidden_size: int = 300,
        num_layers: int = 4,
        angular_dims: Union[Tuple, bool] = tuple(),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_dif = n_dof = len(G.nodes())
        chs = [n_dof * 2 * dof_ndim] + num_layers * [hidden_size]
        self.net = nn.Sequential(
            *[FCswish(chs[i], chs[i + 1]) for i in range(num_layers)],
            nn.Linear(chs[-1], 2 * dof_ndim * n_dof)
        )
        self.nfe = 0
        self.angular_dims = (
            list(range(n_dof * dof_ndim)) if angular_dims is True else angular_dims
        )

    def forward(self, t, z):
        """ Computes a batch of `NxD` time derivatives of the state `z` at time `t`
        Args:
            t: Scalar Tensor of the current time
            z: N x D Tensor of the N different states in D dimensions

        Returns: N x D Tensor of the time derivatives
        """
        assert (t.ndim == 0) and (z.ndim == 2)
        half_D = z.shape[-1] // 2
        theta_mod = (z[..., self.angular_dims] + np.pi) % (2 * np.pi) - np.pi
        not_angular_dims = list(set(range(half_D)) - set(self.angular_dims))
        not_angular_q = z[..., not_angular_dims]
        z_mod = torch.cat([theta_mod, not_angular_q, z[..., half_D:]], dim=-1)
        return self.net(z_mod)

    def integrate(self, z0, ts, tol=1e-4):
        """ Integrates an initial state forward in time according to the learned dynamics

        Args:
            z0: (N x 2 x n_dof x dimensionality of each degree of freedom) sized
                Tensor representing initial state. N is the batch size
            ts: a length T Tensor representing the time points to evaluate at
            tol: integrator tolerance

        Returns: a N x T x 2 x n_dof x d sized Tensor
        """
        assert (z0.ndim == 4) and (ts.ndim == 1)
        N = z0.shape[0]
        zt = odeint(self, z0.reshape(N, -1), ts, rtol=tol, method="rk4")
        zt = zt.permute(1, 0, 2)  # T x N x D -> N x T x D
        return zt.reshape(N, len(ts), *z0.shape[1:])


@export
class DeltaNN(NN):
    def integrate(self, z0, ts):
        """ Integrates an initial state forward in time according to the learned
        dynamics using linear approximations

        Args:
            z0: (N x 2 x n_dof x dimensionality of each degree of freedom) sized
                Tensor representing initial state. N is the batch size
            ts: a length T Tensor representing the time points to evaluate at

        Returns: a N x T x 2 x n_dof x d sized Tensor
        """
        assert (z0.ndim == 4) and (ts.ndim == 1)
        N = z0.shape[0]
        dts = ts[1:] - ts[:-1]
        zts = [z0.reshape(N, -1)]
        for dt in dts:
            zts.append(zts[-1] + dt * self(ts[0], zts[-1]))
        return torch.stack(zts, dim=1).reshape(N, len(ts), *z0.shape[1:])
