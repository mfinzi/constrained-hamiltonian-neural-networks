import torch
import torch.nn as nn
from torchdiffeq import odeint
from lie_conv.lieConv import LieResNet
from lie_conv.lieGroups import Trivial
from biases.models.utils import FCtanh, tril_mask, Linear, Reshape
from biases.dynamics.hamiltonian import (
    EuclideanT,
    ConstrainedHamiltonianDynamics,
)
from biases.systems.rigid_body import rigid_DPhi
from typing import Optional, Tuple, Union
from lie_conv.utils import export, Named


class CH(nn.Module, metaclass=Named):  # abstract constrained Hamiltonian network class
    def __init__(
        self,
        G,
        dof_ndim: Optional[int] = None,
        q_ndim: Optional[int] = None,
        angular_dims: Union[Tuple, bool] = tuple(),
        wgrad=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        if angular_dims != tuple():
            print("CH ignores angular_dims")
        if q_ndim is not None:
            print("CH ignores q_ndim")
        self.G = G
        self.nfe = 0
        self.wgrad = wgrad
        self.n_dof = len(G.nodes)
        self.dof_ndim = 1 if dof_ndim is None else dof_ndim
        self.q_ndim = self.n_dof * self.dof_ndim
        self.dynamics = ConstrainedHamiltonianDynamics(
            self.H, self.DPhi, wgrad=self.wgrad
        )
        self._Minv_net = torch.nn.Parameter(torch.eye(self.n_dof))
        self.register_buffer("_tril_mask", tril_mask(self._Minv_net))
        print("CH currently assumes potential energy depends only on q")
        print("CH currently assumes time independent Hamiltonian")
        print("CH assumes positions q are in Cartesian coordinates")

    @property
    def tril_Minv(self):
        return self._tril_mask * self._Minv_net

    @property
    def Minv(self):
        """Compute the learned inverse mass matrix M^{-1}

        Args:
            q: N x D Tensor representing the position
        """
        lower_triangular = self.tril_Minv
        Minv_mat = lower_triangular @ lower_triangular.T
        return Minv_mat

    def M(self, qdot):
        """Computes the mass matrix times a vector.
        Note that the input must be in Cartesian coordinates
        """
        assert qdot.ndim == 2
        assert qdot.size(-1) == self.n_dof * self.dof_ndim
        qdot = qdot.reshape(-1, self.n_dof, self.dof_ndim)
        lower_diag = self.tril_Minv
        M_qdot = torch.cholesky_solve(qdot, lower_diag.unsqueeze(0), upper=False)
        M_qdot = M_qdot.reshape(-1, self.n_dof * self.dof_ndim)
        return M_qdot

    def H(self, t, z):
        """ Compute the Hamiltonian H(t, x, p)
        Args:
            t: Scalar Tensor representing time
            z: N x D Tensor of the N different states in D dimensions.
                Assumes that z is [x, p] where x is in Cartesian coordinates.

        Returns: Size N Hamiltonian Tensor
        """
        assert (t.ndim == 0) and (z.ndim == 2)
        assert z.size(-1) == 2 * self.n_dof * self.dof_ndim

        x, p = z.chunk(2, dim=-1)
        x = x.reshape(-1, self.n_dof, self.dof_ndim)
        p = p.reshape(-1, self.n_dof, self.dof_ndim)

        T = EuclideanT(p, self.Minv)
        V = self.compute_V(x)
        return T + V

    def DPhi(self, z):
        return rigid_DPhi(self.G, self.Minv[None], z)

    def forward(self, t, z):
        self.nfe += 1
        return self.dynamics(t, z)

    def compute_V(self, x):
        raise NotImplementedError

    def integrate(self, z0, ts, tol=1e-4):
        """ Integrates an initial state forward in time according to the learned Hamiltonian dynamics

        Assumes that z0 = [x0, xdot0] where x0 is in Cartesian coordinates

        Args:
            z0: (N x 2 x n_dof x dof_ndim) sized
                Tensor representing initial state. N is the batch size
            ts: a length T Tensor representing the time points to evaluate at
            tol: integrator tolerance

        Returns: a N x T x 2 x n_dof x d sized Tensor
        """
        assert (z0.ndim == 4) and (ts.ndim == 1)
        assert z0.size(-1) == self.dof_ndim
        assert z0.size(-2) == self.n_dof
        N = z0.size(0)
        z0 = z0.reshape(N, -1)  # -> N x (2 * n_dof * dof_ndim) =: N x D
        x0, xdot0 = z0.chunk(2, dim=-1)
        p0 = self.M(xdot0)

        xp0 = torch.cat([x0, p0], dim=-1)
        xpt = odeint(self, xp0, ts, rtol=tol, method="rk4")
        xpt = xpt.permute(1, 0, 2)  # T x N x D -> N x T x D

        xpt = xpt.reshape(N, len(ts), 2, self.n_dof, self.dof_ndim)
        xt, pt = xpt.chunk(2, dim=-3)
        # TODO: make Minv @ pt faster by L(L^T @ pt)
        vt = self.Minv.matmul(pt)  # Minv [n_dof x n_dof]. pt [N, T, 1, n_dof, dof_ndim]
        xvt = torch.cat([xt, vt], dim=-3)

        return xvt


@export
class CHNN(CH):
    def __init__(
        self,
        G,
        dof_ndim: Optional[int] = None,
        q_ndim: Optional[int] = None,
        angular_dims: Union[Tuple, bool] = tuple(),
        hidden_size: int = 200,
        num_layers=3,
        wgrad=True,
        **kwargs
    ):
        super().__init__(
            G=G,
            dof_ndim=dof_ndim,
            q_ndim=q_ndim,
            angular_dims=angular_dims,
            wgrad=wgrad,
            **kwargs
        )
        n = len(G.nodes())
        chs = [n * self.dof_ndim] + num_layers * [hidden_size]
        self.potential_net = nn.Sequential(
            *[
                FCtanh(chs[i], chs[i + 1], zero_bias=True, orthogonal_init=True)
                for i in range(num_layers)
            ],
            Linear(chs[-1], 1, zero_bias=True, orthogonal_init=True),
            Reshape(-1)
        )

    def compute_V(self, x):
        """ Input is a canonical position variable and the system parameters,
        Args:
            x: (N x n_dof x dof_ndim) sized Tensor representing the position in
            Cartesian coordinates
        Returns: a length N Tensor representing the potential energy
        """
        assert x.ndim == 3
        return self.potential_net(x.reshape(x.size(0), -1))


@export
class CHLC(CH, LieResNet):
    def __init__(
        self,
        G,
        dof_ndim: Optional[int] = None,
        q_ndim: Optional[int] = None,
        angular_dims: Union[Tuple, bool] = tuple(),
        hidden_size=200,
        num_layers=3,
        wgrad=True,
        bn=False,
        group=Trivial(2),
        knn=False,
        nbhd=100,
        mean=True,
        **kwargs
    ):
        n_dof = len(G.nodes())
        super().__init__(
            G=G,
            dof_ndim=dof_ndim,
            q_ndim=q_ndim,
            angular_dims=angular_dims,
            wgrad=wgrad,
            chin=n_dof,
            ds_frac=1,
            num_layers=num_layers,
            nbhd=nbhd,
            mean=mean,
            bn=bn,
            xyz_dim=dof_ndim,
            group=group,
            fill=1.0,
            k=hidden_size,
            num_outputs=1,
            cache=False,
            knn=knn,
            **kwargs
        )

    def compute_V(self, x):
        """ Input is a canonical position variable and the system parameters,
        Args:
            x: (N x n_dof x dof_ndim) sized Tensor representing the position in
            Cartesian coordinates
        Returns: a length N Tensor representing the potential energy
        """
        assert x.ndim == 3
        mask = ~torch.isnan(x[..., 0])
        # features = torch.zeros_like(x[...,:1])
        bs, n, d = x.shape
        features = 1 * torch.eye(n, device=x.device, dtype=x.dtype)[None].repeat(
            (bs, 1, 1)
        )
        return super(CH, self).forward((x, features, mask)).squeeze(-1)
