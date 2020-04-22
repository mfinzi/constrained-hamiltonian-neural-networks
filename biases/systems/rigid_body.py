import torch
import networkx as nx
from torchdiffeq import odeint  # odeint_adjoint as odeint
from oil.utils.utils import Named, export
from biases.animation import Animation
from biases.dynamics.hamiltonian import ConstrainedHamiltonianDynamics, EuclideanT
import numpy as np
from collections import OrderedDict
from scipy.spatial.transform import Rotation

@export
class BodyGraph(nx.Graph):
    """docstring"""
    def __init__(self):
        super().__init__()
        self.key2id = OrderedDict()
    def add_node(self,key,*args,**kwargs):
        #print(key,len(self.key2id),self.key2id)
        self.key2id[key]=len(self.key2id)
        super().add_node(key,*args,**kwargs)

    def add_extended_nd(self,key,m,moments,d=3):
        """ Adds an extended body with name key, mass m and vector of principal
            moments representing the eigenvalues of the the 2nd moment matrix
            along principle directions. 
            d specifies the dimensional extent of the rigid body:
            d=0 is a point mass with 1dof, 
            d=1 is a 1d nodesobject (eg beam) with 2dof
            d=2 is a 2d object (eg plane or disk) with 3dof
            d=3 is a 3d object (eg box,sphere) with 4dof"""
        self.add_node(key,m=m,d=d)
        for i in range(d):
            child_key = f'{key}_{i}'
            self.add_node(child_key)
            self.add_edge(key,child_key,internal=True,l=1.,I=moments[i-1])
            for j in range(i):
                self.add_edge(f'{key}_{j}',child_key,internal=True,l=np.sqrt(2))

    def add_joint(key1,pos1,key2=None,pos2=None):
        """ adds a joint between extended bodies key1 and key2 at the position
            in the body frame 1 pos1 and body frame 2 pos2. pos1 and pos2 should
            be d dimensional vectors, where d is the dimension of the extended body."""
        if key2 is not None:
            self.add_edge(key1,key2,external=True,joint=(pos1,pos2))
        else:
            self.nodes[key1]['joint']=pos1

def edges_wattribute(G,node,attribute):
    all_edges = G.edges(node,data=True)
    return dict((x[:-1], x[-1][attribute]) for x in edges if attribute in x[-1])

@export
class RigidBody(object, metaclass=Named):
    """ Two dimensional rigid body consisting of point masses on nodes (with zero inertia)
        and beams with mass and inertia connecting nodes. Edge inertia is interpreted as
        the unitless quantity, I/ml^2. Ie 1/12 for a beam, 1/2 for a disk"""

    body_graph = NotImplemented
    _m = None
    _minv = None

    def mass_matrix(self):
        """ """
        n = len(self.body_graph.nodes)
        M = torch.zeros(n, n).double()
        for ki, mass in nx.get_node_attributes(self.body_graph, "m").items():
            i = self.body_graph.key2id[ki]
            M[i, i] += mass
        for (ki,kj), I in nx.get_edge_attributes(self.body_graph,"I").items():
            i,j = self.body_graph.key2id[ki],self.body_graph.key2id[kj]
            M[i,i] += I
            M[i,j] -= I
            M[j,i] -= I
            M[j,j] += I
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

    # def global2bodyCoords(self,z):
    #     """ inputs [xv (bs,2,n_all,d)]
    #         outputs [qv (bs,2,D)]"""
    #     # FOR Trees only right now
    #     i = 0
    #     #for key in nx.get_node_attributes(self.body_graph,"root"):

    #     raise NotImplementedError

    # def subtree_global2body_fill(self,node,z,qqdot_out,traversed_nodes,filled_amnt):
    #     """ [z (bs,2,n_all,d)] [qqdot_out (bs,2,D)]"""
    #     traversed_nodes.add(node)
    #     # Deal with internal edges
    #     i = self.body.key2id[node]
    #     zcom = z[:,:,i,:]
    #     cols = []
    #     for edge in edges_wattribute(self.body_graph,node,'internal'):
    #         j = self.body.key2id[edge[0] if edge[0]!=node else edge[1]]
    #         cols.append(z[:,:,j]-zcom)
    #     d_obj = len(cols)
    #     R = torch.stack(cols,dim=-1) # (bs,2,d_ambient,d_obj)

    #     # deal with external edges

    # def body2globalCoords(self):
    #     raise NotImplementedError  # TODO: use nx.bfs_edges and tethers


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


def point2point_constraints(G,x,v):
    """ inputs [Graph] [x (bs,n,d)] [v (bs,n,d)]
        outputs [DPhi (bs,2,n,d,2,C)] """
    bs,n,d = x.shape
    p2p_consrts = nx.get_edge_attributes(G,'l')
    DPhi = torch.zeros(bs, 2, n, d, 2,len(p2p_consrts), device=x.device, dtype=x.dtype)
    for cid,((ki,kj),_) in enumerate(p2p_consrts.items()):
        i,j = G.key2id[ki],G.key2id[kj]
        # Fill out dphi/dx
        DPhi[:, 0,i, :, 0,cid] = 2 * (x[:, i] - x[:, j])
        DPhi[:, 0,j, :, 0,cid] = 2 * (x[:, j] - x[:, i])
        # Fill out d\dot{phi}/dx
        DPhi[:, 0,i, :, 1,cid] = 2 * (v[:, i] - v[:, j])
        DPhi[:, 0,j, :, 1,cid] = 2 * (v[:, j] - v[:, i])
        # Fill out d\dot{phi}/dp
        DPhi[:, 1,i, :, 1,cid] = 2 * (x[:, i] - x[:, j])
        DPhi[:, 1,j, :, 1,cid] = 2 * (x[:, j] - x[:, i])
        #DPhi[:, 1,:, :, 1,cid] = (2 * (x[:, i] - x[:, j])[:, None, :] * (Minv[:, i] - Minv[:, j])[:, :, None])
    return DPhi

def joint_constraints(G,x,v):
    """ inputs [Graph] [x (bs,n,d)] [v (bs,n,d)]
        outputs [DPhi (bs,2,n,d,2,C)].
        Since the constraints are linear, x,v are not required. """
    bs,n,d = x.shape
    edge_joints = nx.get_edge_attributes(G,'joint')
    node_joints = nx.get_node_attributes(G,'joint')
    DPhi = torch.zeros(bs, 2, n, d, 2,len(edge_joints)+len(node_joints),d, device=x.device, dtype=x.dtype)
    for jid, ((ki,kj),(c1,c2)) in enumerate(edge_joints.items()):
        i,j = G.key2id[ki],G.key2id[kj]
        c1t = torch.cat([1-c1.sum()[None],c1])
        c2t = torch.cat([1-c2.sum()[None],c2])
        di = G.nodes[i]['d']
        dj = G.nodes[j]['d']
        for k in range(d):# (bs, di+1, d, d)
            DPhi[:,0,i:1+di,k,0,jid,k] = c1t[None]
            DPhi[:,0,j:1+dj,k,0,jid,k] = -c2t[None]
            DPhi[:,1,i:1+di,k,1,jid,k] = c1t[None]
            DPhi[:,1,j:1+dj,k,1,jid,k] = -c2t[None]
    for jid2, (ki,c1) in enumerate(node_joints.items()):
        i = G.key2id[ki]
        c1t = torch.cat([1-c1.sum()[None],c1])
        di = G.nodes[i]['d']
        for k in range(d):# (bs, di+1, d, d)
            DPhi[:,0,i:1+di,k,0,jid2+jid,k] = c1t[None]
            DPhi[:,1,i:1+di,k,1,jid2+jid,k] = c1t[None]

def point2tether_constraints(G,x,v):
    """ inputs [Graph] [x (bs,n,d)] [v (bs,n,d)]
        outputs [DPhi (bs,2,n,d,2,C)] """
    bs,n,d = x.shape
    tethers = nx.get_node_attributes(G,"tether")
    DPhi = torch.zeros(bs, 2, n, d, 2,len(tethers), device=x.device, dtype=x.dtype)
    for cid, (ki, pos) in enumerate(tethers.items()):
        i = G.key2id[ki]
        ci = pos[None].to(x.device)
        DPhi[:,0, i, :, 0,cid] = 2 * (x[:, i] - ci)
        DPhi[:,0, i, :, 1,cid] = 2 * v[:, i]
        DPhi[:,1, i, :, 1,cid] = 2 * (x[:, i] - ci)
        #DPhi[:,1, :, :, 1,cid] = (2 * (x[:, i] - ci)[:, None, :] * (Minv[:, i])[:, :, None])
    return DPhi

# def axis_constraints(G,x,v,Minv):
#     """ inputs [Graph] [x (bs,n,d)] [v (bs,n,d)]
#         outputs [DPhi (bs,2,n,d,2,C)] """
#     bs,n,d = x.shape
#     axis_constrs = nx.get_node_attributes(G, "pos_cnstr")
#     DPhi = torch.zeros(bs, 2, n, d, 2,len(axis_constrs), device=x.device, dtype=x.dtype)
#     for cid,(i, axis) in enumerate(axis_constrs.items()):
#         DPhi[:,0, i, axis, 0,cid] = 1
#         DPhi[:,0, :, axis, 1,cid] = Minv[:, i]
#     return DPhi

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
    constraints = (point2point_constraints,point2tether_constraints)
    DPhi = torch.cat([constraint(G,x,v) for constraint in constraints],dim=-1)
    DPhi[:,1] = (Minv@DPhi[:,1].reshape(bs,n,-1)).reshape(DPhi[:,1].shape)
    return DPhi.reshape(bs,2*n*d,-1) #(bs,2,n,d,2,C)->#(bs,2nd,2C)
