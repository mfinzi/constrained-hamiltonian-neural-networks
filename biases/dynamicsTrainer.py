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
from biases.lagrangian import LagrangianDynamics
from lie_conv.lieConv import pConvBNrelu, PointConv, Pass, Swish, LieResNet
from lie_conv.lieGroups import Trivial, T
from lie_conv.moleculeTrainer import BottleBlock, GlobalPool
from lie_conv.utils import Expression, export, Named
import numpy as np

from torchdiffeq import odeint
#from torchdiffeq import odeint_adjoint as odeint


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

    def test_rollouts(self,angular_to_euclidean=False,pert_eps=1e-4):
        #self.model.double()
        dataloader = self.dataloaders['test']
        rel_errs = []
        pert_rel_errs = []
        with Eval(self.model), torch.no_grad():
            for mb in dataloader:
                z0,T = mb[0]# assume timesteps evenly spaced for now
                T = T[0]
                dT = (T[-1]-T[0])/len(T)
                long_T = dT*torch.arange(50*len(T)).to(z0.device,z0.dtype)
                zt_pred = self.model.integrate(z0,long_T)
                bs,Nlong,*rest = zt_pred.shape
                # add conversion from angular to euclidean
                body = dataloader.dataset.body
                if angular_to_euclidean:
                    z0 = body.body2globalCoords(z0.squeeze(-1))
                    flat_pred = body.body2globalCoords(zt_pred.reshape(bs*Nlong,*rest).squeeze(-1))
                    zt_pred = flat_pred.reshape(bs,Nlong,*flat_pred.shape[1:])
                zt = dataloader.dataset.body.integrate(z0,long_T)
                perturbation = pert_eps*torch.randn_like(z0)/(zt**2).sum().sqrt()
                zt_pert = dataloader.dataset.body.integrate(z0+perturbation,long_T)
                # (bs,T,2,n,2)
                rel_error = ((zt_pred-zt)**2).sum(-1).sum(-1).sum(-1).sqrt() \
                                /((zt_pred+zt)**2).sum(-1).sum(-1).sum(-1).sqrt()
                rel_errs.append(rel_error)
                pert_rel_error = ((zt_pert-zt)**2).sum(-1).sum(-1).sum(-1).sqrt() \
                                /((zt_pert+zt)**2).sum(-1).sum(-1).sum(-1).sqrt()
                pert_rel_errs.append(pert_rel_error)
            rel_errs = torch.cat(rel_errs,dim=0) # (D,T)
            pert_rel_errs = torch.cat(pert_rel_errs,dim=0) # (D,T)
            both = torch.stack([rel_errs,pert_rel_errs],dim=-1) # (D,T,2)
        return both
                
                




def logspace(a, b, k):
    return np.exp(np.linspace(np.log(a), np.log(b), k))


def FCtanh(chin, chout):
    return nn.Sequential(nn.Linear(chin, chout), nn.Tanh())

def FCswish(chin, chout):
    return nn.Sequential(nn.Linear(chin, chout), Swish())

def FCsoftplus(chin, chout):
    return nn.Sequential(nn.Linear(chin, chout), nn.Softplus())


def tril_mask(square_mat):
    n = square_mat.size(-1)
    coords = square_mat.new(n)
    torch.arange(n, out=coords)
    return coords <= coords.view(n, 1)


from torch.nn.utils import spectral_norm
def add_spectral_norm(module):
    if isinstance(module,  (nn.ConvTranspose1d,
                            nn.ConvTranspose2d,
                            nn.ConvTranspose3d,
                            nn.Conv1d,
                            nn.Conv2d,
                            nn.Conv3d)):
        spectral_norm(module,dim = 1)
        #print("SN on conv layer: ",module)
    elif isinstance(module, nn.Linear):
        spectral_norm(module,dim = 0)
        #print("SN on linear layer: ",module)


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

@export
class LNN(nn.Module,metaclass=Named):
    def __init__(self,G,hidden_size=256,num_layers=4,angular_dims=[],**kwargs):
        super().__init__(**kwargs)
        # Number of function evaluations
        self.nfe = 0
        # Number of degrees of freedom
        self.n = n = len(G.nodes)
        chs = [2*n] + num_layers * [hidden_size]
        print("LNN currently ignores time as an input")
        self.net = nn.Sequential(
            *[FCsoftplus(chs[i], chs[i + 1]) for i in range(num_layers)],
            nn.Linear(chs[-1], 1),
            Reshape(-1)
        )
        self.angular_dims = list(range(n)) if angular_dims==True else angular_dims
    def forward(self, t, z, wgrad=True):
        """ inputs: [t (T,)], [z (bs,2n)]. Outputs: [F (bs,2n)]"""
        self.nfe+=1
        dynamics = LagrangianDynamics(self.L,wgrad=wgrad)
        #print(t.shape)
        #print(z.shape)
        return dynamics(t, z)

    def L(self,t, z):
        """ inputs: [t (T,)], [z (bs,2nd)]. Outputs: [H (bs,)]"""
        D = z.shape[-1]//2
        theta_mod = (z[...,self.angular_dims]+np.pi)%(2*np.pi) - np.pi
        not_angular_dims = list(set(range(D))-set(self.angular_dims))
        not_angular_q = z[...,not_angular_dims]
        p = z[...,D:]
        z_mod = torch.cat([theta_mod,not_angular_q,p],dim=-1)
        return self.net(z_mod) + 1e-3*(p*p).sum(-1)

    def integrate(self, z0, ts, tol=1e-4):
        """ inputs: [z0 (bs,2,n,d)], [ts (T,)]. Outputs: [xvt (bs,T,2,n,d)]"""
        bs = z0.shape[0]
        xvt = odeint(self, z0.reshape(bs,-1), ts, rtol=tol, method="rk4").permute(1, 0, 2)
        return xvt.reshape(bs,len(ts),*z0.shape[1:])

@export
class HNN(nn.Module,metaclass=Named):
    def __init__(self,G,k=150,num_layers=3,canonical=False,angular_dims=[],**kwargs):
        super().__init__(**kwargs)
        self.nfe = 0
        self.n = n = len(G.nodes)
        self.canonical = False
        chs = [n] + num_layers * [k]
        self.potential_net = nn.Sequential(
            *[FCsoftplus(chs[i], chs[i + 1]) for i in range(num_layers)],
            nn.Linear(chs[-1], 1),
            Reshape(-1)
        )
        self.mass_net = nn.Sequential(
            *[FCsoftplus(chs[i], chs[i + 1]) for i in range(num_layers)],
            nn.Linear(chs[-1], n*n),
            Reshape(-1,n,n)
        )
        self.angular_dims = list(range(n)) if angular_dims==True else angular_dims
    def Minv(self,q):
        """ inputs: [q (bs,n,d)]. Outputs: [M^{-1}: (bs,n,k) -> (bs,n,k)]"""
        eps = 1e-4
        q = q.squeeze(-1)
        D = q.shape[-1]
        theta_mod = (q[...,self.angular_dims]+np.pi)%(2*np.pi) - np.pi
        not_angular_dims = list(set(range(D))-set(self.angular_dims))
        not_angular_q = q[...,not_angular_dims]
        q_mod = torch.cat([theta_mod,not_angular_q],dim=-1).reshape(-1, self.n)
        mass_L = self.mass_net(q_mod).reshape(-1,self.n,self.n)
        lower_diag = tril_mask(mass_L)*mass_L
        mask = torch.eye(self.n,device=q.device,dtype=q.dtype)[None]
        Minv = lower_diag@lower_diag.transpose(-2,-1)+eps*mask
        return Minv
    def M(self,q):
        """ inputs: [q (bs,n,d)]. Outputs: [M: (bs,n,k) -> (bs,n,k)]"""
        # eps = 1e-4
        # mass_L = self.mass_net(q.squeeze(-1)).reshape(-1,self.n,self.n)
        # lower_diag = tril_mask(mass_L)*mass_L
        # mask = torch.eye(self.n,device=q.device,dtype=q.dtype)[None]
        return lambda v: torch.solve(v,self.Minv(q))[0]
    def H(self, t, z):
        """ computes the hamiltonian, inputs (bs,2nd), (bs,n,c)"""
        """ inputs: [t (T,)], [z (bs,2nd)]. Outputs: [H (bs,)]"""
        D = z.shape[-1]//2  # of ODE dims, 2*num_particles*space_dim
        bs = z.shape[0]
        q = z[:, : D]
        theta_mod = (q[...,self.angular_dims]+np.pi)%(2*np.pi) - np.pi
        not_angular_dims = list(set(range(D))-set(self.angular_dims))
        not_angular_q = q[...,not_angular_dims]
        q_mod = torch.cat([theta_mod,not_angular_q],dim=-1).reshape(bs, self.n, -1)
        V = self.compute_V(q_mod)
        p = z[:, D:].reshape(bs, self.n, -1)
        Minv = self.Minv(q)
        T = EuclideanT(p, Minv)
        return T + V

    def forward(self, t, z, wgrad=True):
        """ inputs: [t (T,)], [z (bs,2n)]. Outputs: [F (bs,2n)]"""
        self.nfe+=1
        dynamics = HamiltonianDynamics(self.H, wgrad=wgrad)
        return dynamics(t, z)

    def compute_V(self, q):
        """ inputs: [q (bs,n,d)] outputs: [V (bs,)]"""
        return self.potential_net(q.squeeze(-1)).squeeze(-1)

    def integrate(self, z0, ts, tol=1e-4):
        """  """
        """ inputs: [z0 (bs,2,n,d)], [ts (T,)]. Outputs: [xvs (bs,T,2,n,d)]"""
        bs = z0.shape[0]
        p = self.M(z0[:, 0])(z0[:, 1])
        p = z0[:,1] if self.canonical else self.M(z0[:, 0])(z0[:, 1])
        xp = torch.stack([z0[:, 0], p], dim=1).reshape(bs, -1)
        xpt = odeint(self, xp, ts, rtol=tol, method="rk4")
        xps = xpt.permute(1, 0, 2).reshape(bs*len(ts), *z0.shape[1:])
        vs = (self.Minv(xps[:, 0]) @ xps[:, 1])
        xpt = odeint(self, xp, ts, rtol=tol, method="rk4").permute(1, 0, 2)
        xps = xpt.reshape(bs*len(ts), *z0.shape[1:])
        vs = zps[:,1] if self.canonical else (self.Minv(xps[:, 0]) @ xps[:, 1])
        xvs = torch.stack([xps[:, 0], vs], dim=1).reshape(bs,len(ts),*z0.shape[1:])
        return xvs


class CHNN(nn.Module, metaclass=Named):  # abstract Hamiltonian network class
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
        self.nfe+=1
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
class FC(nn.Module, metaclass=Named):
    def __init__(self, G, d=1, k=300, num_layers=4,angular_dims=[], **kwargs):
        super().__init__()
        n = len(G.nodes())
        chs = [n * 2 * d] + num_layers * [k]
        self.net = nn.Sequential(
            *[FCswish(chs[i], chs[i + 1]) for i in range(num_layers)],
            nn.Linear(chs[-1], 2 * d * n)
        )
        self.nfe = 0
        self.angular_dims = list(range(n*d)) if angular_dims==True else angular_dims

    def forward(self, t, z, wgrad=True):
        D = z.shape[-1]//2
        theta_mod = (z[...,self.angular_dims]+np.pi)%(2*np.pi) - np.pi
        not_angular_dims = list(set(range(D))-set(self.angular_dims))
        not_angular_q = z[...,not_angular_dims]
        z_mod = torch.cat([theta_mod,not_angular_q,z[...,D:]],dim=-1)
        return self.net(z_mod)

    def integrate(self, z0, ts, tol=1e-4):
        """ inputs [z0: (bs, z_dim), ts: (bs, T), sys_params: (bs, n, c)]
            outputs pred_zs: (bs, T, z_dim) """
        bs = z0.shape[0]
        zt = odeint(self, z0.reshape(bs,-1), ts, rtol=tol, method="rk4").permute(1, 0, 2)
        return zt.reshape(bs,len(ts),*z0.shape[1:])


@export
class CHFC(CHNN):
    def __init__(self, G, k=150, num_layers=4, d=2):
        super().__init__(G)
        n = len(G.nodes())
        chs = [n * d] + num_layers * [k]
        self.net = nn.Sequential(
            *[FCtanh(chs[i], chs[i + 1]) for i in range(num_layers)],
            nn.Linear(chs[-1], 1)
        )
        #self.apply(add_spectral_norm)

    def compute_V(self, x):
        """ Input is a canonical position variable and the system parameters,
            shapes (bs, n,d) and (bs,n,c)"""
        return self.net(x.reshape(x.shape[0], -1)).squeeze(-1)


@export
class CHLC(CHNN, LieResNet):
    def __init__(self,G,d=2,bn=False,num_layers=4,group=Trivial(2),
                    k=384,knn=False,nbhd=100,mean=True,**kwargs):
        chin = len(G.nodes())
        super().__init__(G=G,chin=chin,ds_frac=1,num_layers=num_layers,
            nbhd=nbhd,mean=mean,bn=bn,xyz_dim=d,group=group,fill=1.0,
            k=k,num_outputs=1,cache=True,knn=knn,**kwargs)
        self.nfe = 0

    def compute_V(self, x):
        """ Input is a canonical position variable and the system parameters,
            shapes (bs, n,d) and (bs,n,c)"""
        mask = ~torch.isnan(x[..., 0])
        # features = torch.zeros_like(x[...,:1])
        bs, n, d = x.shape
        features = torch.eye(n, device=x.device, dtype=x.dtype)[None].repeat((bs, 1, 1))
        return super(CHNN, self).forward((x, features, mask)).squeeze(-1)
