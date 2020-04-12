import torch
import networkx as nx
from torchdiffeq import odeint  # odeint_adjoint as odeint
from oil.utils.utils import Named, export
from biases.animation import Animation


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
    pos_constraints = nx.get_node_attributes(G, "pos_cnstr")
    NC = (
        len(G.edges) + len(tethers) + len(pos_constraints)
    )  # total number of constraints
    v = Minv @ p
    dphi_dx = torch.zeros(bs, n, d, NC, device=z.device, dtype=z.dtype)
    dphi_dp = torch.zeros(bs, n, d, NC, device=z.device, dtype=z.dtype)
    dphid_dx = torch.zeros(bs, n, d, NC, device=z.device, dtype=z.dtype)
    dphid_dp = torch.zeros(bs, n, d, NC, device=z.device, dtype=z.dtype)
    cid = 0  # constraint id
    for e in G.edges:
        i, j = e
        # Fill out dphi/dx
        dphi_dx[:, i, :, cid] = 2 * (x[:, i] - x[:, j])
        dphi_dx[:, j, :, cid] = 2 * (x[:, j] - x[:, i])
        # Fill out d\dot{phi}/dx
        dphid_dx[:, i, :, cid] = 2 * (v[:, i] - v[:, j])
        dphid_dx[:, j, :, cid] = 2 * (v[:, j] - v[:, i])
        # Fill out d\dot{phi}/dp
        dphid_dp[:, :, :, cid] = (
            2 * (x[:, i] - x[:, j])[:, None, :] * (Minv[:, i] - Minv[:, j])[:, :, None]
        )
        cid += 1
    for (i, pos) in tethers.items():
        ci = pos[None].to(x.device)
        dphi_dx[:, i, :, cid] = 2 * (x[:, i] - ci)
        dphid_dx[:, i, :, cid] = 2 * v[:, i]
        dphid_dp[:, :, :, cid] = (
            2 * (x[:, i] - ci)[:, None, :] * (Minv[:, i])[:, :, None]
        )
        cid += 1
    for (i, axis) in pos_constraints.items():
        dphi_dx[:, i, axis, cid] = 1
        dphid_dp[:, :, axis, cid] = Minv[:, i]
        cid += 1
    dPhi_dx = torch.cat(
        [dphi_dx.reshape(bs, n * d, NC), dphid_dx.reshape(bs, n * d, NC)], dim=2
    )
    dPhi_dp = torch.cat(
        [dphi_dp.reshape(bs, n * d, NC), dphid_dp.reshape(bs, n * d, NC)], dim=2
    )
    DPhi = torch.cat([dPhi_dx, dPhi_dp], dim=1)
    return DPhi
