import copy
import torch
import torch.nn as nn
from oil.utils.utils import Eval
from oil.model_trainers import Trainer
from biases.hamiltonian import (
    HamiltonianDynamics,
    ConstrainedHamiltonianDynamics,
    EuclideanT,
    rigid_DPhi,
)
from lie_conv.lieConv import pConvBNrelu, PointConv, Pass, Swish, LieResNet
from lie_conv.lieGroups import Trivial, T
from lie_conv.moleculeTrainer import BottleBlock, GlobalPool
from lie_conv.utils import Expression, export, Named
import numpy as np

# from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint as odeint


@export
class IntegratedDynamicsTrainer(Trainer):
    """ Model should specify the dynamics, mapping from t,z,sysP -> dz/dt"""

    def __init__(self, *args, tol=1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self.hypers["tol"] = tol
        self.num_mbs = 0

    def loss(self, minibatch):
        """ Standard cross-entropy loss """
        (z0, ts), true_zs = minibatch
        pred_zs = self.model.integrate(z0, ts[0], tol=self.hypers["tol"])
        self.num_mbs += 1
        return (pred_zs - true_zs).pow(2).mean()

    def metrics(self, loader):
        mse = lambda mb: self.loss(mb).cpu().data.numpy()
        return {"MSE": self.evalAverageMetrics(loader, mse)}

    def logStuff(self, step, minibatch=None):
        self.logger.add_scalars(
            "info", {"nfe": self.model.nfe / (max(self.num_mbs, 1e-3))}, step
        )
        super().logStuff(step, minibatch)


def logspace(a, b, k):
    return np.exp(np.linspace(np.log(a), np.log(b), k))


def FCswish(chin, chout):
    return nn.Sequential(nn.Linear(chin, chout), Swish())


def tril_mask(square_mat):
    n = square_mat.size(-1)
    coords = square_mat.new(n)
    torch.arange(n, out=coords)
    return coords <= coords.view(n, 1)


class CHNN(nn.Module, metaclass=Named):  # abstract Hamiltonian network class
    def __init__(self, G,constrained=True, **kwargs):
        super().__init__(**kwargs)
        self.G = G
        self.nfe = 0
        self.constrained = constrained
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
        if self.constrained:
            dynamics = ConstrainedHamiltonianDynamics(self.H, self.DPhi, wgrad=wgrad)
        else:
            dynamics = HamiltonianDynamics(self.H,wgrad=wgrad)
        return dynamics(t, z)

    def compute_V(self, q):
        raise NotImplementedError

    def integrate(self, z0, ts, tol=1e-6):
        """ inputs [z0: (bs, z_dim), ts: (bs, T), sys_params: (bs, n, c)]
            outputs pred_zs: (bs, T, z_dim) """
        bs = z0.shape[0]
        xp = torch.stack([z0[:, 0], self.M(z0[:, 1])], dim=1).reshape(bs, -1)
        xpt = odeint(self, xp, ts, rtol=tol, method="rk4")
        xps = xpt.permute(1, 0, 2).reshape(bs, len(ts), *z0.shape[1:])
        xvs = torch.stack([xps[:, :, 0], self.Minv @ xps[:, :, 1]], dim=2)
        return xvs


@export
class FC(nn.Module, metaclass=Named):
    def __init__(self, G, d=2, k=300, num_layers=4, **kwargs):
        super().__init__()
        n = len(G.nodes())
        chs = [n * 2 * d] + num_layers * [k]
        self.net = nn.Sequential(
            *[FCswish(chs[i], chs[i + 1]) for i in range(num_layers)],
            nn.Linear(chs[-1], 2 * d * n)
        )
        self.nfe = 0

    def forward(self, t, z, wgrad=True):
        D = z.shape[-1]
        q = z[:, : D // 2].reshape(*m.shape, -1)
        p = z[:, D // 2 :]
        zm = torch.cat(
            (
                (q - q.mean(1, keepdims=True)).reshape(z.shape[0], -1),
                p,
                sysP.reshape(z.shape[0], -1),
            ),
            dim=1,
        )
        return self.net(zm)

    def integrate(self, z0, ts, rtol=1e-4):
        """ inputs [z0: (bs, z_dim), ts: (bs, T), sys_params: (bs, n, c)]
            outputs pred_zs: (bs, T, z_dim) """
        return odeint(self, z0, ts, rtol=rtol, method="rk4").permute(1, 0, 2)


@export
class CHFC(CHNN):
    def __init__(self, G,constrained=True, k=150, num_layers=4, d=2):
        super().__init__(G,constrained)
        n = len(G.nodes())
        chs = [n * d] + num_layers * [k]
        self.net = nn.Sequential(
            *[FCswish(chs[i], chs[i + 1]) for i in range(num_layers)],
            nn.Linear(chs[-1], 1)
        )

    def compute_V(self, x):
        """ Input is a canonical position variable and the system parameters,
            shapes (bs, n,d) and (bs,n,c)"""
        return self.net(x.reshape(x.shape[0], -1)).squeeze(-1)


@export
class CHLC(CHNN, LieResNet):
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
            cache=True,
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
        features = torch.eye(n, device=x.device, dtype=x.dtype)[None].repeat((bs, 1, 1))
        return super(CHNN, self).forward((x, features, mask)).squeeze(-1)
