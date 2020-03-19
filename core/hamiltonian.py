import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx

class HamiltonianDynamics(nn.Module):
    """ Defines the dynamics given a hamiltonian. If wgrad=True, the dynamics can be backproped."""
    def __init__(self,H,wgrad=False):
        super().__init__()
        self.H = H
        self.wgrad=wgrad
        self.nfe=0
    def forward(self,t,z):
        self.nfe+=1
        with torch.enable_grad():
            z = torch.zeros_like(z, requires_grad=True) + z
            D = z.shape[-1]
            h = self.H(t,z).sum() # elements in mb are independent, gives mb gradients
            rg = torch.autograd.grad(h,z,create_graph=self.wgrad)[0] # riemannian gradient
        sg = torch.cat([rg[:,D//2:],-rg[:,:D//2]],dim=-1) # symplectic gradient = SdH
        return sg

class ConstrainedHamiltonianDynamics(nn.Module):
    """ Defines the dynamics given a hamiltonian. If wgrad=True, the dynamics can be backproped."""
    def __init__(self,H,DPhi,wgrad=False):
        super().__init__()
        self.H = H
        self.DPhi = DPhi
        self.wgrad=wgrad
        self.nfe=0
    def forward(self,t,z):
        self.nfe+=1
        with torch.enable_grad():
            z = torch.zeros_like(z, requires_grad=True) + z
            P = Proj(self.DPhi(z))
            H = self.H(t,z).sum() # elements in mb are independent, gives mb gradients
            dH = torch.autograd.grad(H,z,create_graph=self.wgrad)[0] # riemannian gradient
        return P(J(dH.unsqueeze(-1))).squeeze(-1)

def rigid_DPhi(rigid_body_graph,Minv,z):
    """inputs [Graph (n,E)] [x (bs,n,d)] [p (bs,n,d)] [Minv (bs, n, n)]
       ouput [DPhi (bs, 2nd, 2E)]"""
    bs, n = Minv.shape[:2]
    D = z.shape[-1] # of ODE dims, 2*num_particles*space_dim
    x = z[:,:D//2].reshape(bs,n,-1)
    p = z[:,D//2:].reshape(bs,n,-1)
    bs,n,d = x.shape
    G = rigid_body_graph
    tethers = nx.get_node_attributes(G,'tether')
    E = len(G.edges)+ len(tethers)
    v = Minv@p
    dphi_dx = torch.zeros(bs,n,d,E)
    dphi_dp = torch.zeros(bs,n,d,E)
    dphid_dx = torch.zeros(bs,n,d,E)
    dphid_dp = torch.zeros(bs,n,d,E)
    for i,e in enumerate(G.edges):
        n,m = e
        # Fill out dphi/dx
        dphi_dx[:,n,:,i] =  2*(x[:,n]-x[:,m])
        dphi_dx[:,m,:,i] =  2*(x[:,m]-x[:,n])
        # Fill out d\dot{phi}/dx
        dphid_dx[:,n,:,i] = 2*(v[:,n] - v[:,m])
        dphid_dx[:,m,:,i] = 2*(v[:,m] - v[:,n])
        # Fill out d\dot{phi}/dp
        dphid_dp[:,:,:,i] = 2*(x[:,n]-x[:,m])[:,None,:]*(Minv[:,n] - Minv[:,m])[:,:,None]
    for i,(n,pos) in enumerate(tethers.items()):
        cn = pos[None].to(x.device)
        dphi_dx[:,n,:,i+n] =  2*(x[:,n]-cn)
        dphid_dx[:,n,:,i+n] = 2*v[:,n]
        dphid_dp[:,:,:,i+n] = 2*(x[:,n]-cn)[:,None,:]*(Minv[:,n])[:,:,None]
    dPhi_dx = torch.cat([dphi_dx.reshape(bs,n*d,E), dphid_dx.reshape(bs,n*d,E)],dim=1)
    dPhi_dp = torch.cat([dphi_dp.reshape(bs,n*d,E), dphid_dp.reshape(bs,n*d,E)],dim=1)
    DPhi = torch.cat([dPhi_dx,dPhi_dp],dim=2)
    return DPhi

def J(M):
    """ applies the J matrix to another matrix M.
        input: M (*,2nd,b), output: J@M (*,2nd,b)"""
    star = M.shape[:-2]
    b = M.shape[-1]
    Mqp = M.reshape(*star,2,-1,b)
    JM = torch.cat([-Mqp[...,1,:,:],Mqp[...,0,:,:]],dim=-2)
    return JM

def Proj(DPhi):
    def _P(M):
        DPhiT = DPhi.transpose(-1,-2)
        X,_ = torch.solve(DPhiT@J(DPhi),DPhiT@M)
        return M - J(DPhi@X)
    return _P
        
def EuclideanT(p, Minv):
    """ Shape (bs,n,d), and (bs,n,n),
        standard \sum_n pT Minv p/2 kinetic energy"""
    return (p*(Minv@p)).sum(-1).sum(-1)


class RigidBody(object):
    self.body_graph = NotImplemented
    def mass_matrix(self):
        n = len(self.body_graph.nodes)
        M = torch.zeros(n,n)
        for i, mass in nx.get_node_attributes(self.body_graph,'m').items():
            M[i,i] += mass
        for (i,j), mass in nx.get_edge_attributes(self.body_graph,'m').items():
            M[i,i] += mass
            M[i,j] += mass
            M[j,i] += mass
            M[j,j] += mass
        for (i,j), inertia in nx.get_edge_attributes(self.body_graph,'I').items():
            M[i,i] += inertia
            M[i,j] -= inertia
            M[j,i] -= inertia
            M[j,j] += inertia
        return M
    def global2bodyCoords(self):
        raise NotImplementedError
    def body2globalCoords(self):
        raise NotImplementedError
    def sample_initial_conditions(self,n_systems):
        raise NotImplementedError


class ChainPendulum(RigidBody):
    def __init__(self,links=2,beams=False,m=1,l=1):
        self.body_graph = nx.Graph()
        if beams:
            self.body_graph.add_node(0,tether = torch.zeros(2))
            for i in range(1,links):
                self.body_graph.add_node(i)
                self.body_graph.add_edge(i-1,i,m=m,I=m/12,l=l)
        else:
            self.body_graph.add_node(0,m=m,tether = torch.zeros(2))
            for i in range(1,links):
                self.body_graph.add_node(i,m=m)
                self.body_graph.add_edge(i-1,i,l=l)
# Make animation plots look nicer. Why are there leftover points on the trails?
class Animation2d(object):
    def __init__(self, qt, ms=None, box_lim=(-1, 1)):
        if ms is None: ms = len(qt)*[6]
        self.qt = qt
        self.fig = plt.figure()
        self.ax = self.fig.add_axes([0, 0, 1, 1])#axes(projection='3d')
        self.ax.set_xlim(box_lim)
        self.ax.set_ylim(box_lim)
        self.lines = sum([self.ax.plot([],[],'-') for particle in self.qt],[])
        self.pts = sum([self.ax.plot([],[],'o',ms=ms[i]) for i in range(len(self.qt))],[])
    def init(self):
        for line,pt in zip(self.lines,self.pts):
            line.set_data([], [])
            pt.set_data([], [])
        return self.lines + self.pts
    def update(self,i=0):
        for line, pt, trajectory in zip(self.lines,self.pts,self.qt):
            x,y = trajectory[:,:i]
            line.set_data(x,y)
            pt.set_data(x[-1:], y[-1:])
        #self.fig.clear()
        self.fig.canvas.draw()
        return self.lines+self.pts
    def animate(self):
        return animation.FuncAnimation(self.fig,self.update,frames=self.qt.shape[-1],
                                       interval=33,init_func=self.init,blit=True)