import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import functools
from oil.utils.utils import Named, export, Expression
from torchdiffeq import odeint#odeint_adjoint as odeint
from biases.animation import Animation

class HamiltonianDynamics(nn.Module):
    """ Defines the dynamics given a hamiltonian. If wgrad=True, the dynamics can be backproped."""

    def __init__(self, H, wgrad=False):
        super().__init__()
        self.H = H
        self.wgrad = wgrad
        self.nfe = 0

    def forward(self, t, z):
        self.nfe += 1
        with torch.enable_grad():
            z = torch.zeros_like(z, requires_grad=True) + z
            H = self.H(t, z).sum()  # elements in mb are independent, gives mb gradients
            dH = torch.autograd.grad(H, z, create_graph=self.wgrad)[0]  # gradient
        return J(dH.unsqueeze(-1)).squeeze(-1)


class ConstrainedHamiltonianDynamics(nn.Module):
    """ Defines the dynamics given a hamiltonian. If wgrad=True, the dynamics can be backproped."""

    def __init__(self, H, DPhi, wgrad=False):
        super().__init__()
        self.H = H
        self.DPhi = DPhi
        self.wgrad = wgrad
        self.nfe = 0

    def forward(self, t, z):
        self.nfe += 1
        with torch.enable_grad():
            z = torch.zeros_like(z, requires_grad=True) + z
            P = Proj(self.DPhi(z))
            H = self.H(t, z).sum()  # elements in mb are independent, gives mb gradients
            dH = torch.autograd.grad(H, z, create_graph=self.wgrad)[0]  # gradient
        return P(J(dH.unsqueeze(-1))).squeeze(-1)


def rigid_DPhi(rigid_body_graph, Minv, z):
    """inputs [Graph (n,E)] [x (bs,n,d)] [p (bs,n,d)] [Minv (bs, n, n)]
       ouput [DPhi (bs, 2nd, 2E)]"""
    n = Minv.shape[-1]
    bs, D = z.shape  # of ODE dims, 2*num_particles*space_dim
    x = z[:, : D // 2].reshape(bs, n, -1)
    p = z[:, D // 2 :].reshape(bs, n, -1)
    bs, n, d = x.shape

    G = rigid_body_graph
    tethers = nx.get_node_attributes(G, "tether")
    pos_constraints = nx.get_node_attributes(G,'pos_cnstr')
    NC = len(G.edges) + len(tethers) + len(pos_constraints) # total number of constraints
    v = Minv @ p
    dphi_dx = torch.zeros(bs, n, d, NC, device=z.device, dtype=z.dtype)
    dphi_dp = torch.zeros(bs, n, d, NC, device=z.device, dtype=z.dtype)
    dphid_dx = torch.zeros(bs, n, d, NC, device=z.device, dtype=z.dtype)
    dphid_dp = torch.zeros(bs, n, d, NC, device=z.device, dtype=z.dtype)
    cid=0 #constraint id
    for e in G.edges:
        i, j = e
        # Fill out dphi/dx
        dphi_dx[:, i, :, cid] = 2 * (x[:, i] - x[:, j])
        dphi_dx[:, j, :, cid] = 2 * (x[:, j] - x[:, i])
        # Fill out d\dot{phi}/dx
        dphid_dx[:, i, :, cid] = 2 * (v[:, i] - v[:, j])
        dphid_dx[:, j, :, cid] = 2 * (v[:, j] - v[:, i])
        # Fill out d\dot{phi}/dp
        dphid_dp[:, :, :, cid] = 2*(x[:,i] - x[:,j])[:,None,:]*(Minv[:,i]-Minv[:,j])[:,:,None]
        cid +=1
    for (i, pos) in tethers.items():
        ci = pos[None].to(x.device)
        dphi_dx[:, i, :, cid] = 2 * (x[:, i] - ci)
        dphid_dx[:, i, :, cid] = 2 * v[:, i]
        dphid_dp[:, :, :, cid] = 2 * (x[:, i] - ci)[:, None, :] * (Minv[:, i])[:, :, None]
        cid +=1
    for (i,axis) in pos_constraints.items():
        dphi_dx[:, i, axis, cid] = 1
        dphid_dp[:,:,axis,cid] = Minv[:, i]
        cid +=1
    dPhi_dx = torch.cat([dphi_dx.reshape(bs, n * d, NC), dphid_dx.reshape(bs, n * d, NC)], dim=2)
    dPhi_dp = torch.cat([dphi_dp.reshape(bs, n * d, NC), dphid_dp.reshape(bs, n * d, NC)], dim=2)
    DPhi = torch.cat([dPhi_dx, dPhi_dp], dim=1)
    return DPhi


def J(M):
    """ applies the J matrix to another matrix M.
        input: M (*,2nd,b), output: J@M (*,2nd,b)"""
    *star, D, b = M.shape
    JM = torch.cat([M[..., D // 2 :, :], -M[..., : D // 2, :]], dim=-2)
    return JM


def Proj(DPhi):
    def _P(M):
        DPhiT = DPhi.transpose(-1, -2)
        reg = 0  # 1e-4*torch.eye(DPhi.shape[-1],dtype=DPhi.dtype,device=DPhi.device)[None]
        X, _ = torch.solve(DPhiT @ M, DPhiT @ J(DPhi) + reg)
        return M - J(DPhi @ X)
    return _P


def EuclideanT(p, Minv, function=False):
    """ Shape (bs,n,d), and (bs,n,n),
        standard \sum_n pT Minv p/2 kinetic energy"""
    if function:
        return (p * Minv(p)).sum(-1).sum(-1) / 2
    else:
        return (p * (Minv @ p)).sum(-1).sum(-1) / 2

@export
class RigidBody(object, metaclass=Named):
    """ Two dimensional rigid body consisting of point masses on nodes (with zero inertia)
        and beams with mass and inertia connecting nodes. Edge inertia is interpreted as 
        the unitless quantity, I/ml^2. Ie 1/12 for a beam, 1/2 for a disk"""

    body_graph = NotImplemented
    _m = None
    _minv = None

    def mass_matrix(self):
        """ For mass and inertia on edges, we assume the center of mass
            of the segment is the midpoint between (x_i,x_j): x_com = (x_i+x_j)/2"""
        n = len(self.body_graph.nodes)
        M = torch.zeros(n, n).double()
        for i, mass in nx.get_node_attributes(self.body_graph, "m").items():
            M[i, i] += mass
        for (i, j), mass in nx.get_edge_attributes(self.body_graph, "m").items():
            M[i, i] += mass / 4
            M[i, j] += mass / 4
            M[j, i] += mass / 4
            M[j, j] += mass / 4
        for (i, j), inertia in nx.get_edge_attributes(self.body_graph, "I").items():
            M[i, i] += inertia * mass
            M[i, j] -= inertia * mass
            M[j, i] -= inertia * mass
            M[j, j] += inertia * mass
        return M

    @property
    def M(self):
        if self._m is None:
            self._m = self.mass_matrix()
        return self._m

    @property
    def Minv(self):
        if self._minv is None:
            self._minv = self.M.inverse()
        return self._minv

    def to(self, device=None, dtype=None):
        self._m = self._m.to(device, dtype)
        self._minv = self._minv.to(device, dtype)

    def DPhi(self, z):
        Minv = self.Minv[None].to(device=z.device, dtype=z.dtype)
        return rigid_DPhi(self.body_graph, Minv, z)

    def global2bodyCoords(self):
        raise NotImplementedError

    def body2globalCoords(self):
        raise NotImplementedError  # TODO: use nx.bfs_edges and tethers

    def sample_initial_conditions(self, n_systems):
        raise NotImplementedError

    def potential(self, x):
        raise NotImplementedError

    def hamiltonian(self, t, z):
        bs, D = z.shape  # of ODE dims, 2*num_particles*space_dim
        n = len(self.body_graph.nodes)
        x = z[:, : D // 2].reshape(bs, n, -1)
        p = z[:, D // 2 :].reshape(bs, n, -1)
        T = EuclideanT(p, self.Minv)
        V = self.potential(x)
        return T + V

    def dynamics(self, wgrad=False):
        return ConstrainedHamiltonianDynamics(self.hamiltonian, self.DPhi, wgrad=wgrad)

    def integrate(self, z0, T, tol=1e-5):  # (x,v) -> (x,p) -> (x,v)
        """ Integrate system from z0 to times in T (e.g. linspace(0,10,100))"""
        bs = z0.shape[0]
        xp = torch.stack(
            [z0[:, 0].double(), self.M @ z0[:, 1].double()], dim=1
        ).reshape(bs, -1)
        with torch.no_grad():
            xpt = odeint(self.dynamics(), xp, T.double(), rtol=tol, method="dopri5")
        xps = xpt.permute(1, 0, 2).reshape(bs, len(T), *z0.shape[1:])
        xvs = torch.stack([xps[:, :, 0], self.Minv.double() @ xps[:, :, 1]], dim=2)
        return xvs.to(z0.device)

    def animate(self,zt):
        # bs, T, 2,n,d
        if len(zt.shape)==5:
            j = np.random.randint(zt.shape[0])
            xt = zt[j,:,0,:,:]
        else:
            xt = zt[:,0,:,:]
        anim = self.animator(xt,self)
        return anim.animate()
    @property
    def animator(self):
        return Animation

def jvp(y, x, v):
    with torch.enable_grad():
        Jv = torch.autograd.grad(y, [x], [v])[0]
    return Jv


def vjp(y, x, v):
    # Following the trick from https://j-towns.github.io/2017/06/12/A-new-trick.html
    with torch.enable_grad():
        u = torch.ones_like(
            y, requires_grad=True
        )  # Dummy variable (could take any value)
        Ju = torch.autograd.grad(y, [x], [u], create_graph=True)[0]
        vJ = torch.autograd.grad(Ju, [u], [v])[0]
    return vJ


import numpy as np


def MLE(xt, ts, F, v0=None):
    """ Computes the Maximal Lyapunov exponent using the Power iteration.
        inputs: trajectory [xt (T,*)] dynamics [F] """
    v = torch.randn_like(xt[0]) if v0 is None else v0
    dt = ts[1] - ts[0]
    exps = []
    for i, x in enumerate(xt):
        # for j in range(5):
        x = torch.zeros_like(x, requires_grad=True) + x.detach()
        y = F(ts[i], x[None])[0]
        u = v + vjp(y, x, v).detach() * dt
        # u  = v+ dt*(F(ts[i],(x+1e-7*v)[None])[0]-F(ts[i],x[None])[0])/(1e-7)
        r = (u ** 2).sum().sqrt().detach()
        v = u / r  # P((u/r)[None,:,None])[0,:,0]
        exps += [r.log().item() / dt]  # (1/i)*(r.log() - exp)
        # print(r.log()/(100/5000))
    return np.array(exps)  # ,u


class LyapunovDynamics(nn.Module):
    def __init__(self, F):
        super().__init__()
        self.F = F

    def forward(self, t, xqr):
        n = (xqr.shape[-1] - 1) // 2
        with torch.enable_grad():
            x = xqr[..., :n] + torch.zeros_like(xqr[..., :n], requires_grad=True)
            q = xqr[..., n : 2 * n]
            xdot = self.F(t, x)
            DFq = vjp(xdot, x, q)
        qDFq = (q * DFq).sum(-1, keepdims=True)
        qdot = DFq - q * qDFq
        lrdot = qDFq
        xqrdot = torch.cat([xdot, qdot, lrdot], dim=-1)
        return xqrdot


def MLE2(x0, F, ts, **kwargs):
    with torch.no_grad():
        LD = LyapunovDynamics(F)
        x0 = x0.reshape(x0.shape[0], -1)
        q0 = torch.randn_like(x0)
        q0 /= (q0 ** 2).sum(-1, keepdims=True).sqrt()
        lr0 = torch.zeros(x0.shape[0], 1, dtype=x0.dtype, device=x0.device)
        Lx0 = torch.cat([x0, q0, lr0], dim=-1)
        Lxt = odeint(LD, Lx0, ts, **kwargs)
        maximal_exponent = Lxt  # [...,-1]
    return maximal_exponent