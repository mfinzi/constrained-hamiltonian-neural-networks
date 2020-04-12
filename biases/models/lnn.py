import torch
from torch import Tensor
import torch.nn as nn
from torchdiffeq import odeint
from lie_conv.utils import export, Named
from biases.models.utils import FCswish, Reshape
from biases.dynamics.lagrangian import LagrangianDynamics
import numpy as np
from typing import Tuple, Union


@export
class LNN(nn.Module, metaclass=Named):
    def __init__(
        self,
        G,
        hidden_size: int = 256,
        num_layers: int = 4,
        angular_dims: Union[Tuple, bool] = tuple(),
        wgrad: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        # Number of function evaluations
        self.nfe = 0
        # Number of degrees of freedom
        self.n_dof = n_dof = len(G.nodes)
        chs = [2 * n_dof] + num_layers * [hidden_size]
        self.net = nn.Sequential(
            *[FCswish(chs[i], chs[i + 1]) for i in range(num_layers)],
            nn.Linear(chs[-1], 1),
            Reshape(-1)
        )
        print("LNN currently assumes time independent Lagrangian")
        self.angular_dims = (
            tuple(range(n_dof)) if angular_dims is True else angular_dims
        )
        self.dynamics = LagrangianDynamics(self.L, wgrad=wgrad)

    def forward(self, t: Tensor, z: Tensor):
        """ Computes a batch of `NxD` time derivatives of the state `z` at time `t`
        Args:
            t: Scalar Tensor of the current time
            z: N x D Tensor of the N different states in D dimensions

        Returns: N x D Tensor of the time derivatives
        """
        assert (t.ndim == 0) and (z.ndim == 2)
        self.nfe += 1
        return self.dynamics(t, z)

    def L(self, t: Tensor, z: Tensor):
        """ Compute the Lagrangian L(t, q, qdot)
        Args:
            t: Scalar Tensor representing time
            z: N x D Tensor of the N different states in D dimensions.
                Assumes that z is [q, qdot]

        Returns: Size N Lagrangian Tensor
        """
        assert (t.ndim == 0) and (z.ndim == 2)
        d = z.shape[-1] // 2
        theta_mod = (z[..., self.angular_dims] + np.pi) % (2 * np.pi) - np.pi
        not_angular_dims = list(set(range(d)) - set(self.angular_dims))
        not_angular_q = z[..., not_angular_dims]
        qdot = z[..., d:]
        z_mod = torch.cat([theta_mod, not_angular_q, qdot], dim=-1)
        # TODO: why + 1e-1?
        return self.net(z_mod) + 1e-1 * (qdot * qdot).sum(-1)

    def integrate(self, z0: Tensor, ts: Tensor, tol=1e-4) -> Tensor:
        """ Integrates an initial state forward in time according to the learned Lagrangian dynamics

        Args:
            z0: (N x 2 x n_dof x dimensionality of each degree of freedom) sized
                Tensor representing initial state. N is the batch size
            ts: a length T Tensor representing the time points to evaluate at
            tol: integrator tolerance

        Returns: a N x T x 2 x n_dof x d sized Tensor
        """
        assert (z0.ndim == 4) and (ts.ndim == 1)
        N = z0.shape[0]
        xvt = odeint(self, z0.reshape(N, -1), ts, rtol=tol, method="rk4")
        xvt = xvt.permute(1, 0, 2)  # T x N x D -> N x T x D
        return xvt.reshape(N, len(ts), *z0.shape[1:])
