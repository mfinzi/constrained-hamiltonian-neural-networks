import torch
import torch.nn as nn
from torchdiffeq import odeint
from lie_conv.utils import export, Named
from biases.models.utils import FCsoftplus, Reshape, tril_mask
from biases.dynamics.hamiltonian import HamiltonianDynamics, EuclideanT
import numpy as np


@export
class HNN(nn.Module, metaclass=Named):
    def __init__(
        self, G, k=150, num_layers=3, canonical=False, angular_dims=[], **kwargs
    ):
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
            nn.Linear(chs[-1], n * n),
            Reshape(-1, n, n)
        )
        self.angular_dims = list(range(n)) if angular_dims == True else angular_dims

    def Minv(self, q):
        """ inputs: [q (bs,n,d)]. Outputs: [M^{-1}: (bs,n,k) -> (bs,n,k)]"""
        eps = 1e-4
        q = q.squeeze(-1)
        D = q.shape[-1]
        theta_mod = (q[..., self.angular_dims] + np.pi) % (2 * np.pi) - np.pi
        not_angular_dims = list(set(range(D)) - set(self.angular_dims))
        not_angular_q = q[..., not_angular_dims]
        q_mod = torch.cat([theta_mod, not_angular_q], dim=-1).reshape(-1, self.n)
        mass_L = self.mass_net(q_mod).reshape(-1, self.n, self.n)
        lower_diag = tril_mask(mass_L) * mass_L
        mask = torch.eye(self.n, device=q.device, dtype=q.dtype)[None]
        Minv = lower_diag @ lower_diag.transpose(-2, -1) + eps * mask
        return Minv

    def M(self, q):
        """ inputs: [q (bs,n,d)]. Outputs: [M: (bs,n,k) -> (bs,n,k)]"""
        # eps = 1e-4
        # mass_L = self.mass_net(q.squeeze(-1)).reshape(-1,self.n,self.n)
        # lower_diag = tril_mask(mass_L)*mass_L
        # mask = torch.eye(self.n,device=q.device,dtype=q.dtype)[None]
        return lambda v: torch.solve(v, self.Minv(q))[0]

    def H(self, t, z):
        """ computes the hamiltonian, inputs (bs,2nd), (bs,n,c)"""
        """ inputs: [t (T,)], [z (bs,2nd)]. Outputs: [H (bs,)]"""
        D = z.shape[-1] // 2  # of ODE dims, 2*num_particles*space_dim
        bs = z.shape[0]
        q = z[:, :D]
        theta_mod = (q[..., self.angular_dims] + np.pi) % (2 * np.pi) - np.pi
        not_angular_dims = list(set(range(D)) - set(self.angular_dims))
        not_angular_q = q[..., not_angular_dims]
        q_mod = torch.cat([theta_mod, not_angular_q], dim=-1).reshape(bs, self.n, -1)
        V = self.compute_V(q_mod)
        p = z[:, D:].reshape(bs, self.n, -1)
        Minv = self.Minv(q)
        T = EuclideanT(p, Minv)
        return T + V

    def forward(self, t, z, wgrad=True):
        """ inputs: [t (T,)], [z (bs,2n)]. Outputs: [F (bs,2n)]"""
        self.nfe += 1
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
        p = z0[:, 1] if self.canonical else self.M(z0[:, 0])(z0[:, 1])
        xp = torch.stack([z0[:, 0], p], dim=1).reshape(bs, -1)
        xpt = odeint(self, xp, ts, rtol=tol, method="rk4")
        xps = xpt.permute(1, 0, 2).reshape(bs * len(ts), *z0.shape[1:])
        vs = self.Minv(xps[:, 0]) @ xps[:, 1]
        xpt = odeint(self, xp, ts, rtol=tol, method="rk4").permute(1, 0, 2)
        xps = xpt.reshape(bs * len(ts), *z0.shape[1:])
        vs = xps[:, 1] if self.canonical else (self.Minv(xps[:, 0]) @ xps[:, 1])
        xvs = torch.stack([xps[:, 0], vs], dim=1).reshape(bs, len(ts), *z0.shape[1:])
        return xvs
