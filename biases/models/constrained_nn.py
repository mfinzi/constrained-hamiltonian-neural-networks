import torch
import torch.nn as nn
from torchdiffeq import odeint
from oil.utils.utils import export, Named
from lie_conv.lieConv import LieResNet
from lie_conv.lieGroups import Trivial
from biases.models.utils import FCtanh, tril_mask
from biases.dynamics.hamiltonian import (
    EuclideanT,
    ConstrainedHamiltonianDynamics,
    rigid_DPhi,
)


class CH(nn.Module, metaclass=Named):  # abstract Hamiltonian network class
    def __init__(self, G, **kwargs):
        super().__init__(**kwargs)
        self.G = G
        self.nfe = 0
        self.n = len(self.G.nodes())
        self._m_lower = torch.nn.Parameter(torch.eye(self.n))

    @property
    def Minv(self):
        lower_diag = tril_mask(self._m_lower) * self._m_lower
        reg = torch.eye(self.n, device=lower_diag.device, dtype=lower_diag.dtype)
        return lower_diag @ lower_diag.T  # +1e-4*reg

    # @property
    def M(self, x):
        lower_diag = tril_mask(self._m_lower) * self._m_lower
        Mx = torch.cholesky_solve(x, lower_diag)
        return Mx  # - 1e-4*torch.cholesky_solve(Mx,lower_diag)

    def H(self, t, z):
        """ computes the hamiltonian, inputs (bs,2nd), (bs,n,c)"""
        D = z.shape[-1]  # of ODE dims, 2*num_particles*space_dim
        Minv = self.Minv
        bs = z.shape[0]
        q = z[:, : D // 2].reshape(bs, self.n, -1)
        p = z[:, D // 2 :].reshape(bs, self.n, -1)
        T = EuclideanT(p, Minv)
        V = self.compute_V(q)
        return T + V

    def DPhi(self, z):
        return rigid_DPhi(self.G, self.Minv[None], z)

    def forward(self, t, z, wgrad=True):
        self.nfe += 1
        dynamics = ConstrainedHamiltonianDynamics(self.H, self.DPhi, wgrad=wgrad)
        return dynamics(t, z)

    def compute_V(self, q):
        raise NotImplementedError

    def integrate(self, z0, ts, tol=1e-4):
        """ inputs [z0: (bs, z_dim), ts: (bs, T), sys_params: (bs, n, c)]
            outputs pred_zs: (bs, T, z_dim) """
        bs = z0.shape[0]
        xp = torch.stack([z0[:, 0], self.M(z0[:, 1])], dim=1).reshape(bs, -1)
        xpt = odeint(self, xp, ts, rtol=tol, method="rk4")
        xps = xpt.permute(1, 0, 2).reshape(bs, len(ts), *z0.shape[1:])
        xvs = torch.stack([xps[:, :, 0], self.Minv @ xps[:, :, 1]], dim=2)
        return xvs


@export
class CHNN(CH):
    def __init__(self, G, k=150, num_layers=4, d=2):
        super().__init__(G)
        n = len(G.nodes())
        chs = [n * d] + num_layers * [k]
        self.net = nn.Sequential(
            *[FCtanh(chs[i], chs[i + 1]) for i in range(num_layers)],
            nn.Linear(chs[-1], 1)
        )
        # self.apply(add_spectral_norm)

    def compute_V(self, x):
        """ Input is a canonical position variable and the system parameters,
            shapes (bs, n,d) and (bs,n,c)"""
        return self.net(x.reshape(x.shape[0], -1)).squeeze(-1)


@export
class CHLC(CH, LieResNet):
    def __init__(
        self,
        G,
        d=2,
        bn=False,
        num_layers=4,
        group=Trivial(2),
        k=384,
        knn=False,
        nbhd=100,
        mean=True,
        **kwargs
    ):
        chin = len(G.nodes())
        super().__init__(
            G=G,
            chin=chin,
            ds_frac=1,
            num_layers=num_layers,
            nbhd=nbhd,
            mean=mean,
            bn=bn,
            xyz_dim=d,
            group=group,
            fill=1.0,
            k=k,
            num_outputs=1,
            cache=False,
            knn=knn,
            **kwargs
        )
        self.nfe = 0

    def compute_V(self, x):
        """ Input is a canonical position variable and the system parameters,
            shapes (bs, n,d) and (bs,n,c)"""
        mask = ~torch.isnan(x[..., 0])
        # features = torch.zeros_like(x[...,:1])
        bs, n, d = x.shape
        features = 1 * torch.eye(n, device=x.device, dtype=x.dtype)[None].repeat(
            (bs, 1, 1)
        )
        return super(CH, self).forward((x, features, mask)).squeeze(-1)
