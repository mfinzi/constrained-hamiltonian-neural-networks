import torch
import networkx as nx
from torchdiffeq import odeint  # odeint_adjoint as odeint
from oil.utils.utils import Named, export
from biases.animation import Animation
from biases.dynamics.hamiltonian import ConstrainedHamiltonianDynamics, EuclideanT
import numpy as np


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

    def animate(self, zt):
        # bs, T, 2,n,d
        if len(zt.shape) == 5:
            j = np.random.randint(zt.shape[0])
            xt = zt[j, :, 0, :, :]
        else:
            xt = zt[:, 0, :, :]
        anim = self.animator(xt, self)
        return anim.animate()

    @property
    def animator(self):
        return Animation

def point2point_constraints(G,x,v,Minv):
    """ inputs [Graph] [x (bs,n,d)] [v (bs,n,d)]
        outputs [DPhi (bs,2,n,d,2,C)] """
    bs,n,d = x.shape
    DPhi = torch.zeros(bs, 2, n, d, 2,len(G.edges), device=x.device, dtype=x.dtype)
    for cid,(i,j) in enumerate(G.edges):
        # Fill out dphi/dx
        DPhi[:, 0,i, :, 0,cid] = 2 * (x[:, i] - x[:, j])
        DPhi[:, 0,j, :, 0,cid] = 2 * (x[:, j] - x[:, i])
        # Fill out d\dot{phi}/dx
        DPhi[:, 0,i, :, 1,cid] = 2 * (v[:, i] - v[:, j])
        DPhi[:, 0,j, :, 1,cid] = 2 * (v[:, j] - v[:, i])
        # Fill out d\dot{phi}/dp
        DPhi[:, 1,:, :, 1,cid] = (2 * (x[:, i] - x[:, j])[:, None, :] * (Minv[:, i] - Minv[:, j])[:, :, None])
    return DPhi

def point2tether_constraints(G,x,v,Minv):
    """ inputs [Graph] [x (bs,n,d)] [v (bs,n,d)]
        outputs [DPhi (bs,2,n,d,2,C)] """
    bs,n,d = x.shape
    tethers = nx.get_node_attributes(G,"tether")
    DPhi = torch.zeros(bs, 2, n, d, 2,len(tethers), device=x.device, dtype=x.dtype)
    for cid, (i, pos) in enumerate(tethers.items()):
        ci = pos[None].to(x.device)
        DPhi[:,0, i, :, 0,cid] = 2 * (x[:, i] - ci)
        DPhi[:,0, i, :, 1,cid] = 2 * v[:, i]
        DPhi[:,1, :, :, 1,cid] = (2 * (x[:, i] - ci)[:, None, :] * (Minv[:, i])[:, :, None])
    return DPhi

def axis_constraints(G,x,v,Minv):
    """ inputs [Graph] [x (bs,n,d)] [v (bs,n,d)]
        outputs [DPhi (bs,2,n,d,2,C)] """
    bs,n,d = x.shape
    axis_constrs = nx.get_node_attributes(G, "pos_cnstr")
    DPhi = torch.zeros(bs, 2, n, d, 2,len(axis_constrs), device=x.device, dtype=x.dtype)
    for cid,(i, axis) in enumerate(axis_constrs.items()):
        DPhi[:,0, i, axis, 0,cid] = 1
        DPhi[:,0, :, axis, 1,cid] = Minv[:, i]
    return DPhi

def rigid_DPhi(rigid_body_graph, Minv, z):
    """inputs [Graph (n,E)] [z (bs,2nd)] [Minv (bs, n, n)]
       ouput [DPhi (bs, 2nd, 2C)]"""
    n = Minv.shape[-1]
    bs, D = z.shape  # of ODE dims, 2*num_particles*space_dim
    x = z[:, : D // 2].reshape(bs, n, -1)
    p = z[:, D // 2 :].reshape(bs, n, -1)
    bs, n, d = x.shape
    G = rigid_body_graph
    v = Minv @ p
    constraints = (point2point_constraints,point2tether_constraints,axis_constraints)
    DPhi = torch.cat([constraint(G,x,v,Minv) for constraint in constraints],dim=-1) 
    return DPhi.reshape(bs,2*n*d,-1) #(bs,2,n,d,2,C)->#(bs,2nd,2C)
