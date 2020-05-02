import torch
import torch.nn as nn
from torchdiffeq import odeint
from oil.utils.utils import export, Named
from biases.models.utils import FCsoftplus, mod_angles, Linear
from typing import Tuple, Union


@export
class NN(nn.Module, metaclass=Named):
    def __init__(
        self,
        G,
        dof_ndim: int = 1,
        hidden_size: int = 200,
        num_layers: int = 3,
        angular_dims: Tuple = tuple(),
        wgrad: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        if wgrad:
            print("NN ignores wgrad")
        self.q_ndim = dof_ndim

        chs = [2 * self.q_ndim] + num_layers * [hidden_size]
        self.net = nn.Sequential(
            *[
                FCsoftplus(chs[i], chs[i + 1], zero_bias=True, orthogonal_init=True)
                for i in range(num_layers)
            ],
            Linear(chs[-1], 2 * self.q_ndim, zero_bias=True, orthogonal_init=True)
        )
        print("NN currently assumes time independent ODE")
        self.nfe = 0
        self.angular_dims = angular_dims

    def forward(self, t, z):
        """ Computes a batch of `NxD` time derivatives of the state `z` at time `t`
        Args:
            t: Scalar Tensor of the current time
            z: N x 2D Tensor of the N different states in D dimensions

        Returns: N x 2D Tensor of the time derivatives
        """
        assert (t.ndim == 0) and (z.ndim == 2)
        assert z.size(-1) == 2 * self.q_ndim
        q, qdot = z.chunk(2, dim=-1)
        q_mod = mod_angles(q, self.angular_dims)
        z_mod = torch.cat([q_mod, qdot], dim=-1)
        return self.net(z_mod)

    def integrate(self, z0, ts, tol=1e-4):
        """ Integrates an initial state forward in time according to the learned dynamics

        Args:
            z0: (bs x 2 x D) sized
                Tensor representing initial state. N is the batch size
            ts: a length T Tensor representing the time points to evaluate at
            tol: integrator tolerance

        Returns: a bs x T x 2 x D sized Tensor
        """
        assert (z0.ndim == 3) and (ts.ndim == 1)
        bs = z0.shape[0]
        zt = odeint(self, z0.reshape(bs, -1), ts, rtol=tol, method="rk4")
        zt = zt.permute(1, 0, 2)  # T x N x D -> N x T x D
        return zt.reshape(bs, len(ts), *z0.shape[1:])


@export
class DeltaNN(NN):
    def integrate(self, z0, ts, tol=0.0):
        """ Integrates an initial state forward in time according to the learned
        dynamics using Euler's method with predicted time derivatives

        Args:
            z0: (bs x 2 x D) sized
                Tensor representing initial state. N is the batch size
            ts: a length T Tensor representing the time points to evaluate at

        Returns: a bs x T x 2 x D sized Tensor
        """
        assert (z0.ndim == 3) and (ts.ndim == 1)
        bs = z0.shape[0]
        dts = ts[1:] - ts[:-1]
        zts = [z0.reshape(bs, -1)]
        for dt in dts:
            zts.append(zts[-1] + dt * self(ts[0], zts[-1]))
        return torch.stack(zts, dim=1).reshape(N, len(ts), *z0.shape[1:])
