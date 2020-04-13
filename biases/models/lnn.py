import torch
from torch import Tensor
import torch.nn as nn
from torchdiffeq import odeint
from oil.utils.utils import export, Named
from biases.models.utils import FCswish, Reshape, mod_angles
from biases.dynamics.lagrangian import LagrangianDynamics
from typing import Tuple, Union, Optional


@export
class LNN(nn.Module, metaclass=Named):
    def __init__(
        self,
        G,
        q_ndim: Optional[int] = None,
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
        q_ndim = q_ndim if q_ndim is not None else len(G.nodes)
        self.q_ndim = q_ndim
        chs = [2 * q_ndim] + num_layers * [hidden_size]
        self.net = nn.Sequential(
            *[FCswish(chs[i], chs[i + 1]) for i in range(num_layers)],
            nn.Linear(chs[-1], 1),
            Reshape(-1)
        )
        print("LNN currently assumes time independent Lagrangian")
        # Set everything to angular if `angular_dim` is True
        self.angular_dims = (
            tuple(range(q_ndim)) if angular_dims is True else angular_dims
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
        ret = self.dynamics(t, z)
        self.nfe += 1
        return ret

    def L(self, t: Tensor, z: Tensor, eps=1e-1):
        """ Compute the Lagrangian L(t, q, qdot)
        Args:
            t: Scalar Tensor representing time
            z: N x D Tensor of the N different states in D dimensions.
                Assumes that z is [q, qdot]

        Returns: Size N Lagrangian Tensor
        """
        assert (t.ndim == 0) and (z.ndim == 2)
        assert z.size(-1) == 2 * self.q_ndim
        q, qdot = z.chunk(2, dim=-1)
        q_mod = mod_angles(q, self.angular_dims)
        z_mod = torch.cat([q_mod, qdot], dim=-1)
        # Add regularization to prevent singular mass matrix at initialization
        # equivalent to adding eps to the diagonal of the mass Hessian
        # Note that the network could learn to offset this added term
        reg = eps * (qdot * qdot).sum(-1)
        return self.net(z_mod) + reg

    def integrate(self, z0: Tensor, ts: Tensor, tol=1e-4) -> Tensor:
        """ Integrates an initial state forward in time according to the learned Lagrangian dynamics

        Note that self.q_ndim == n_dof x dimensionality of each degree of freedom

        Args:
            z0: (N x 2 x n_dof x dimensionality of each degree of freedom) sized
                Tensor representing initial state. N is the batch size
            ts: a length T Tensor representing the time points to evaluate at
            tol: integrator tolerance

        Returns: a N x T x 2 x n_dof x d sized Tensor
        """
        assert (z0.ndim == 4) and (ts.ndim == 1)
        assert z0.size(-1) * z0.size(-2) == self.q_ndim
        N = z0.shape[0]
        xvt = odeint(self, z0.reshape(N, -1), ts, rtol=tol, method="rk4")
        xvt = xvt.permute(1, 0, 2)  # T x N x D -> N x T x D
        return xvt.reshape(N, len(ts), *z0.shape[1:])
