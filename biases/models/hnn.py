import torch
from torch import Tensor
import torch.nn as nn
from torchdiffeq import odeint
from oil.utils.utils import export, Named
from biases.models.utils import FCsoftplus, Reshape, tril_mask, mod_angles, Linear
from biases.dynamics.hamiltonian import HamiltonianDynamics, GeneralizedT
from typing import Tuple, Union, Optional


@export
class HNN(nn.Module, metaclass=Named):
    def __init__(
        self,
        G,
        dof_ndim: Optional[int] = None,
        q_ndim: Optional[int] = None,
        hidden_size: int = 200,
        num_layers: int = 3,
        canonical: bool = False,
        angular_dims: Union[Tuple, bool] = tuple(),
        wgrad: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.nfe = 0
        if dof_ndim is not None:
            print("HNN ignores dof_ndim")
        q_ndim = q_ndim if q_ndim is not None else len(G.nodes)
        self.q_ndim = q_ndim
        self.canonical = canonical

        chs = [q_ndim] + num_layers * [hidden_size]
        self.potential_net = nn.Sequential(
            *[
                FCsoftplus(chs[i], chs[i + 1], zero_bias=True, orthogonal_init=True)
                for i in range(num_layers)
            ],
            Linear(chs[-1], 1, zero_bias=True, orthogonal_init=True),
            Reshape(-1)
        )
        print("HNN currently assumes potential energy depends only on q")
        print("HNN currently assumes time independent Hamiltonian")

        self.mass_net = nn.Sequential(
            *[
                FCsoftplus(chs[i], chs[i + 1], zero_bias=True, orthogonal_init=True)
                for i in range(num_layers)
            ],
            Linear(chs[-1], q_ndim * q_ndim, zero_bias=True, orthogonal_init=True),
            Reshape(-1, q_ndim, q_ndim)
        )
        self.register_buffer("_tril_mask", tril_mask(torch.eye(q_ndim)))
        # Set everything to angular if `angular_dim` is True
        self.angular_dims = (
            list(range(q_ndim)) if angular_dims is True else angular_dims
        )
        self.dynamics = HamiltonianDynamics(self.H, wgrad=wgrad)

    def H(self, t, z):
        """ Compute the Hamiltonian H(t, q, p)
        Args:
            t: Scalar Tensor representing time
            z: N x D Tensor of the N different states in D dimensions.
                Assumes that z is [q, p].

        Returns: Size N Hamiltonian Tensor
        """
        assert (t.ndim == 0) and (z.ndim == 2)
        assert z.size(-1) == 2 * self.q_ndim
        q, p = z.chunk(2, dim=-1)
        q_mod = mod_angles(q, self.angular_dims)

        V = self.potential_net(q_mod)

        Minv = self.Minv(q_mod)
        # TODO: should this be p?
        T = GeneralizedT(p, Minv)
        return T + V

    def Minv(self, q: Tensor, eps=1e-1) -> Tensor:
        """Compute the learned inverse mass matrix M^{-1}(q)

        Args:
            q: N x D Tensor representing the position
        """
        assert q.ndim == 2
        lower_triangular = self._tril_mask * self.mass_net(q)
        assert lower_triangular.ndim == 3
        Minv = lower_triangular.matmul(lower_triangular.transpose(-2, -1))
        Minv = Minv + eps * torch.eye(
            Minv.size(-1), device=Minv.device, dtype=Minv.dtype
        )
        return Minv

    def M(self, q):
        """Returns a function that multiplies the mass matrix M(q) by a vector qdot

        Args:
            q: N x D Tensor representing the position
        """
        assert q.ndim == 2
        lower_triangular = self._tril_mask * self.mass_net(q)
        assert lower_triangular.ndim == 3

        def M_func(qdot):
            assert qdot.ndim == 2
            qdot = qdot.unsqueeze(-1)
            # M_times_qdot = torch.cholesky_solve(
            #    qdot, lower_triangular, upper=False
            # ).squeeze(-1)
            M_times_qdot = torch.solve(qdot, self.Minv(q))[0].squeeze(-1)
            return M_times_qdot

        return M_func

    def forward(self, t, z):
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
        assert z0.size(-1) * z0.size(-2) == self.q_ndim
        N, _, n_dof, d = z0.size()
        assert n_dof * d == self.q_ndim
        z0 = z0.reshape(N, -1)  # -> N x D
        if self.canonical:
            q0, p0 = z0.chunk(2, dim=-1)
        else:
            q0, v0 = z0.chunk(2, dim=-1)
            p0 = self.M(q0)(v0)

        qp0 = torch.cat([q0, p0], dim=-1)
        qpt = odeint(self, qp0, ts, rtol=tol, method="rk4")
        qpt = qpt.permute(1, 0, 2)  # T x N x D -> N x T x D

        if self.canonical:
            qpt = qpt.reshape(N, len(ts), 2, n_dof, d)
            return qpt
        else:
            qt, pt = qpt.reshape(-1, 2 * self.q_ndim).chunk(2, dim=-1)
            vt = self.Minv(qt).matmul(pt.unsqueeze(-1)).squeeze(-1)
            qvt = torch.cat([qt, vt], dim=-1)
            qvt = qvt.reshape(N, len(ts), 2, n_dof, d)
            return qvt
