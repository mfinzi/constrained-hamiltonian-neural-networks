import torch
from torch import Tensor
import torch.nn as nn
from torchdiffeq import odeint
from oil.utils.utils import export, Named
from biases.models.utils import FCsoftplus, Reshape, Linear, CosSin
from biases.dynamics.lagrangian import LagrangianDynamics,DeLanDynamics
from typing import Tuple, Union
from  biases.models.hnn import HNN

@export
class LNN(nn.Module, metaclass=Named):
    def __init__(
        self,
        G,
        dof_ndim: int = 1,
        hidden_size: int = 200,
        num_layers: int = 3,
        angular_dims: Tuple = tuple(),
        wgrad: bool = True,
        delan: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        # Number of function evaluations
        self.nfe = 0

        self.q_ndim = dof_ndim

        # We parameterize angular dims in terms of cos(theta), sin(theta)
        if not delan:
            chs = [2 * self.q_ndim + len(angular_dims)] + num_layers * [hidden_size]
            self.net = nn.Sequential(
                CosSin(self.q_ndim, angular_dims, only_q=False),
                *[
                    FCsoftplus(chs[i], chs[i + 1], zero_bias=True, orthogonal_init=True)
                    for i in range(num_layers)
                ],
                Linear(chs[-1], 1, zero_bias=True, orthogonal_init=True),
                Reshape(-1)
            )
        print("LNN currently assumes time independent Lagrangian")
        self.angular_dims = angular_dims
        self.dynamics = LagrangianDynamics(self.L, wgrad=wgrad)

    def forward(self, t: Tensor, z: Tensor):
        """ Computes a batch of `NxD` time derivatives of the state `z` at time `t`
        Args:
            t: Scalar Tensor of the current time
            z: N x 2D Tensor of the N different states in D dimensions

        Returns: N x 2D Tensor of the time derivatives
        """
        assert (t.ndim == 0) and (z.ndim == 2)
        ret = self.dynamics(t, z)
        self.nfe += 1
        return ret

    def L(self, t: Tensor, z: Tensor, eps=1e-1):
        """ Compute the Lagrangian L(t, q, qdot)
        Args:
            t: Scalar Tensor representing time
            z: N x 2D Tensor of the N different states in 2D dimensions.
                Assumes that z is [q, qdot]

        Returns: Size N Lagrangian Tensor
        """
        assert (t.ndim == 0) and (z.ndim == 2)
        assert z.size(-1) == 2 * self.q_ndim
        q, qdot = z.chunk(2, dim=-1)
        # Add regularization to prevent singular mass matrix at initialization
        # equivalent to adding eps to the diagonal of the mass matrix (Hessian of L)
        # Note that the network could learn to offset this added term
        reg = eps * (qdot * qdot).sum(-1)
        return self.net(z) + reg

    def integrate(self, z0: Tensor, ts: Tensor, tol=1e-4, method="rk4") -> Tensor:
        """ Integrates an initial state forward in time according to the learned Lagrangian dynamics

        Note that self.q_ndim == n_dof x dimensionality of each degree of freedom

        Args:
            z0: (N x 2 x D) sized
                Tensor representing initial state. N is the batch size
            ts: a length T Tensor representing the time points to evaluate at
            tol: integrator tolerance

        Returns: a N x T x 2 x D sized Tensor
        """
        assert (z0.ndim == 3) and (ts.ndim == 1)
        #assert z0.size(-1) * z0.size(-2) == self.q_ndim
        assert z0.shape[-1] == self.q_ndim
        bs = z0.shape[0]
        self.nfe = 0
        xvt = odeint(self, z0.reshape(bs, -1), ts, rtol=tol, method=method)
        xvt = xvt.permute(1, 0, 2)  # T x N x D -> N x T x D
        return xvt.reshape(bs, len(ts), *z0.shape[1:])


# @export
# class DeLaN(LNN):
#     def __init__(
#         self,
#         G,
#         dof_ndim: int = 1,
#         hidden_size: int = 200,
#         num_layers: int = 3,
#         angular_dims: Tuple = tuple(),
#         wgrad: bool = True,
#         **kwargs
#     ):
#         super().__init__(G=G, dof_ndim=dof_ndim, hidden_size=hidden_size,
#                 num_layers=num_layers, angular_dims=angular_dims, wgrad=wgrad,
#                 delan=True, **kwargs)
#         self.nfe = 0

#         self.q_ndim = dof_ndim
#         self.angular_dims = angular_dims

#         # We parameterize angular dims in terms of cos(theta), sin(theta)
#         chs = [self.q_ndim + len(angular_dims)] + num_layers * [hidden_size]
#         self.potential_net = nn.Sequential(
#             CosSin(self.q_ndim, angular_dims, only_q=True),
#             *[
#                 FCsoftplus(chs[i], chs[i + 1], zero_bias=True, orthogonal_init=True)
#                 for i in range(num_layers)
#             ],
#             Linear(chs[-1], 1, zero_bias=True, orthogonal_init=True),
#             Reshape(-1)
#         )
#         print("HNN currently assumes potential energy depends only on q")
#         print("HNN currently assumes time independent Hamiltonian")

#         self.mass_net = nn.Sequential(
#             CosSin(self.q_ndim, angular_dims, only_q=True),
#             *[
#                 FCsoftplus(chs[i], chs[i + 1], zero_bias=True, orthogonal_init=True)
#                 for i in range(num_layers)
#             ],
#             Linear(
#                 chs[-1], self.q_ndim * self.q_ndim, zero_bias=True, orthogonal_init=True
#             ),
#             Reshape(-1, self.q_ndim, self.q_ndim)
#         )
#         self.dynamics = LagrangianDynamics(self.L, wgrad=wgrad)

#     def net(self, z, eps=1e-1):
#         assert z.size(-1) == 2 * self.q_ndim
#         q, qdot = z.chunk(2, dim=-1)

#         V = self.potential_net(q)
#         M = self.mass_net(q)
#         reg = eps * (qdot * qdot).sum(-1)
#         T = (qdot * (M @ qdot.unsqueeze(-1)).squeeze(-1)).sum(-1) / 2.0 + reg
#         return T - V


@export
class DeLaN(HNN):
    """ The new DeLaN that uses a native implementation of DeLaNdynamics for increased performance"""
    def __init__(self,*args,wgrad=True,**kwargs):
        super().__init__(*args,wgrad=True,**kwargs)
        M = lambda q,v: self.M(q)(v)
        Minv = lambda q,p: (self.Minv(q)@p.unsqueeze(-1)).squeeze(-1)
        self.dynamics = DeLanDynamics(self.potential_net,M,Minv, wgrad=wgrad)

    def forward(self, t: Tensor, z: Tensor):
        """ Computes a batch of `NxD` time derivatives of the state `z` at time `t`
        Args:
            t: Scalar Tensor of the current time
            z: N x 2D Tensor of the N different states in D dimensions

        Returns: N x 2D Tensor of the time derivatives
        """
        assert (t.ndim == 0) and (z.ndim == 2)
        ret = self.dynamics(t, z)
        self.nfe += 1
        return ret

    def integrate(self, z0: Tensor, ts: Tensor, tol=1e-4, method="rk4") -> Tensor:
        """ Integrates an initial state forward in time according to the learned Lagrangian dynamics

        Note that self.q_ndim == n_dof x dimensionality of each degree of freedom

        Args:
            z0: (N x 2 x D) sized
                Tensor representing initial state. N is the batch size
            ts: a length T Tensor representing the time points to evaluate at
            tol: integrator tolerance

        Returns: a N x T x 2 x D sized Tensor
        """
        assert (z0.ndim == 3) and (ts.ndim == 1)
        #assert z0.size(-1) * z0.size(-2) == self.q_ndim
        assert z0.shape[-1] == self.q_ndim
        bs = z0.shape[0]
        self.nfe = 0
        xvt = odeint(self, z0.reshape(bs, -1), ts, rtol=tol, method=method)
        xvt = xvt.permute(1, 0, 2)  # T x N x D -> N x T x D
        return xvt.reshape(bs, len(ts), *z0.shape[1:])