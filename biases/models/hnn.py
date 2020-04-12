import torch
from torch import Tensor
import torch.nn as nn
from torchdiffeq import odeint
from lie_conv.utils import export, Named
from biases.models.utils import FCsoftplus, Reshape, tril_mask
from biases.dynamics.hamiltonian import HamiltonianDynamics, EuclideanT
import numpy as np
from typing import Tuple, Union


@export
class HNN(nn.Module, metaclass=Named):
    def __init__(
        self,
        G,
        hidden_size: int = 150,
        num_layers: int = 3,
        canonical: bool = False,
        angular_dims: Union[Tuple, bool] = tuple(),
        wgrad: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.nfe = 0
        self.n_dof = n_dof = len(G.nodes)
        self.canonical = canonical
        chs = [n_dof] + num_layers * [hidden_size]
        self.potential_net = nn.Sequential(
            *[FCsoftplus(chs[i], chs[i + 1]) for i in range(num_layers)],
            nn.Linear(chs[-1], 1),
            Reshape(-1)
        )
        print("HNN currently assumes time independent Hamiltonian")
        self.mass_net = nn.Sequential(
            *[FCsoftplus(chs[i], chs[i + 1]) for i in range(num_layers)],
            nn.Linear(chs[-1], n_dof * n_dof),
            Reshape(-1, n_dof, n_dof)  # Here we assume each dof is 1 dimensional
        )
        self.angular_dims = list(range(n_dof)) if angular_dims is True else angular_dims
        self.dynamics = HamiltonianDynamics(self.H, wgrad=wgrad)

    def Minv(self, q: Tensor, eps: int = 1e-4) -> Tensor:
        """Compute the learned inverse mass matrix M^{-1}

        Args:
            q: N x n_dof x D Tensor representing the position
            eps: diagonal noise to add to M^{-1}
        """
        assert q.ndim == 3
        N = q.size(0)
        q = q.reshape(N, -1)
        d = q.shape[-1]  # n_dof x D

        theta_mod = (q[..., self.angular_dims] + np.pi) % (2 * np.pi) - np.pi
        not_angular_dims = list(set(range(d)) - set(self.angular_dims))
        not_angular_q = q[..., not_angular_dims]
        q_mod = torch.cat([theta_mod, not_angular_q], dim=-1).reshape(N, -1)

        mass_L = self.mass_net(q_mod)
        lower_diag = tril_mask(mass_L) * mass_L
        mask = torch.eye(mass_L.size(-1), device=q.device, dtype=q.dtype)
        Minv = lower_diag @ lower_diag.transpose(-2, -1) + eps * mask
        return Minv

    def M(self, q):
        """Returns a function that multiplies the mass matrix M by a vector v

        Args:
            q: N x n_dof x D Tensor representing the position
            eps: diagonal noise to add to M^{-1}
        """
        return lambda v: torch.solve(v, self.Minv(q))[0]

    def H(self, t, z):
        """ Compute the Hamiltonian H(t, q, v or p)
        Args:
            t: Scalar Tensor representing time
            z: N x D Tensor of the N different states in D dimensions.
                Assumes that z is [q, v or p].
                If self.canonical is True then we assume p instead of v

        Returns: Size N Hamiltonian Tensor
        """
        if self.canonical:
            raise NotImplementedError
        # TODO: factor out the theta mod preprocessing
        assert (t.ndim == 0) and (z.ndim == 2)
        half_D = (
            z.shape[-1] // 2
        )  # half the number of ODE dims, i.e.  num_particles*space_dim
        N = z.shape[0]
        q = z[:, :half_D]
        theta_mod = (q[..., self.angular_dims] + np.pi) % (2 * np.pi) - np.pi
        not_angular_dims = list(set(range(half_D)) - set(self.angular_dims))
        not_angular_q = q[..., not_angular_dims]
        q_mod = torch.cat([theta_mod, not_angular_q], dim=-1)
        V = self.potential_net(q_mod)
        p = z[:, half_D:].reshape(N, self.n_dof, -1)
        Minv = self.Minv(q)
        T = EuclideanT(p, Minv)
        return T + V

    def forward(self, t, z):
        """ Computes a batch of `NxD` time derivatives of the state `z` at time `t`
        Args:
            t: Scalar Tensor of the current time
            z: N x D Tensor of the N different states in D dimensions

        Returns: N x D Tensor of the time derivatives
        """
        assert (t.ndim == 0) and (z.ndim == 2)
        self.nfe += 1
        return self.dynamics(t, z)

    def integrate(self, z0, ts, tol=1e-4):
        """ Integrates an initial state forward in time according to the learned Hamiltonian dynamics

        Args:
            z0: (N x 2 x n_dof x dimensionality of each degree of freedom) sized
                Tensor representing initial state. N is the batch size
            ts: a length T Tensor representing the time points to evaluate at
            tol: integrator tolerance

        Returns: a N x T x 2 x n_dof x d sized Tensor
        """
        assert (z0.ndim == 4) and (ts.ndim == 1)
        N = z0.shape[0]
        if self.canonical:
            q0, p0 = z0.chunk(2, dim=1)
        else:
            q0, v0 = z0.chunk(2, dim=1)
            p0 = self.M(q0)(v0)

        qp0 = torch.stack([q0, p0], dim=1).reshape(N, -1)
        qpt = odeint(self, qp0, ts, rtol=tol, method="rk4")
        qpt = qpt.permute(1, 0, 2)  # T x N x D -> N x T x D

        if self.canonical:
            qpt = qpt.reshape(N, len(ts), *z0.shape[1:])
            return qpt
        else:
            qt, pt = qpt.chunk(2, dim=-1)
            vt = self.Minv(qt) @ pt
            qvt = torch.cat([qt, vt], dim=-1)
            qvt = qvt.reshape(N, len(ts), *z0.shape[1:])
            return qvt
