import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import functools
from oil.utils.utils import Named, export, Expression
from torchdiffeq import odeint_adjoint as odeint

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
    n = Minv.shape[-1]
    bs, D = z.shape # of ODE dims, 2*num_particles*space_dim
    x = z[:,:D//2].reshape(bs,n,-1)
    p = z[:,D//2:].reshape(bs,n,-1)
    bs,n,d = x.shape
    
    G = rigid_body_graph
    tethers = nx.get_node_attributes(G,'tether')
    E = len(G.edges)
    ET = E + len(tethers)
    v = Minv@p
    dphi_dx = torch.zeros(bs,n,d,ET,device=z.device,dtype=z.dtype)
    dphi_dp = torch.zeros(bs,n,d,ET,device=z.device,dtype=z.dtype)
    dphid_dx = torch.zeros(bs,n,d,ET,device=z.device,dtype=z.dtype)
    dphid_dp = torch.zeros(bs,n,d,ET,device=z.device,dtype=z.dtype)
    for eid,e in enumerate(G.edges):
        i,j = e
        # Fill out dphi/dx
        dphi_dx[:,i,:,eid] =  2*(x[:,i]-x[:,j])
        dphi_dx[:,j,:,eid] =  2*(x[:,j]-x[:,i])
        # Fill out d\dot{phi}/dx
        dphid_dx[:,i,:,eid] = 2*(v[:,i] - v[:,j])
        dphid_dx[:,j,:,eid] = 2*(v[:,j] - v[:,i])
        # Fill out d\dot{phi}/dp
        dphid_dp[:,:,:,eid] = 2*(x[:,i]-x[:,j])[:,None,:]*(Minv[:,i] - Minv[:,j])[:,:,None]
    for vid,(i,pos) in enumerate(tethers.items()):
        ci = pos[None].to(x.device)
        dphi_dx[:,i,:,vid+E] =  2*(x[:,i]-ci)
        dphid_dx[:,i,:,vid+E] = 2*v[:,i]
        dphid_dp[:,:,:,vid+E] = 2*(x[:,i]-ci)[:,None,:]*(Minv[:,i])[:,:,None]
    dPhi_dx = torch.cat([dphi_dx.reshape(bs,n*d,ET), dphid_dx.reshape(bs,n*d,ET)],dim=2)
    dPhi_dp = torch.cat([dphi_dp.reshape(bs,n*d,ET), dphid_dp.reshape(bs,n*d,ET)],dim=2)
    DPhi = torch.cat([dPhi_dx,dPhi_dp],dim=1)
    return DPhi

def J(M):
    """ applies the J matrix to another matrix M.
        input: M (*,2nd,b), output: J@M (*,2nd,b)"""
    *star,D,b = M.shape
    JM = torch.cat([M[...,D//2:,:],-M[...,:D//2,:]],dim=-2)
    return JM

def Proj(DPhi):
    def _P(M):
        DPhiT = DPhi.transpose(-1,-2)
        reg = 0#1e-4*torch.eye(DPhi.shape[-1],dtype=DPhi.dtype,device=DPhi.device)[None]
        X,_ = torch.solve(DPhiT@M,DPhiT@J(DPhi)+reg)
        return M - J(DPhi@X)
    return _P
        
def EuclideanT(p, Minv):
    """ Shape (bs,n,d), and (bs,n,n),
        standard \sum_n pT Minv p/2 kinetic energy"""
    return (p*(Minv@p)).sum(-1).sum(-1)/2


class RigidBody(object,metaclass=Named):
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
        M = torch.zeros(n,n).double()
        for i, mass in nx.get_node_attributes(self.body_graph,'m').items():
            M[i,i] += mass
        for (i,j), mass in nx.get_edge_attributes(self.body_graph,'m').items():
            M[i,i] += mass/4
            M[i,j] += mass/4
            M[j,i] += mass/4
            M[j,j] += mass/4
        for (i,j), inertia in nx.get_edge_attributes(self.body_graph,'I').items():
            M[i,i] += inertia*mass
            M[i,j] -= inertia*mass
            M[j,i] -= inertia*mass
            M[j,j] += inertia*mass
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
    def to(self,device=None,dtype=None):
        self._m = self._m.to(device,dtype)
        self._minv = self._minv.to(device,dtype)
    def DPhi(self,z):
        Minv = self.Minv[None].to(device=z.device,dtype=z.dtype)
        return rigid_DPhi(self.body_graph,Minv,z)
    def global2bodyCoords(self):
        raise NotImplementedError
    def body2globalCoords(self):
        raise NotImplementedError #TODO: use nx.bfs_edges and tethers
    def sample_initial_conditions(self,n_systems):
        raise NotImplementedError
    def potential(self,x):
        raise NotImplementedError
    def hamiltonian(self,t,z):
        bs,D = z.shape # of ODE dims, 2*num_particles*space_dim
        n = len(self.body_graph.nodes)
        x = z[:,:D//2].reshape(bs,n,-1)
        p = z[:,D//2:].reshape(bs,n,-1)
        T=EuclideanT(p,self.Minv)
        V = self.potential(x)
        return T+V
    def dynamics(self,wgrad=False):
        return ConstrainedHamiltonianDynamics(self.hamiltonian,self.DPhi,wgrad=wgrad)
    def integrate(self,z0,T,tol=1e-5):# (x,v) -> (x,p) -> (x,v)
        """ Integrate system from z0 to times in T (e.g. linspace(0,10,100))"""
        bs = z0.shape[0]
        xp = torch.stack([z0[:,0].double(),self.M@z0[:,1].double()],dim=1).reshape(bs,-1)
        with torch.no_grad():
            xpt = odeint(self.dynamics(),xp,T.double(),rtol=tol,method='dopri5')
        xps = xpt.permute(1,0,2).reshape(bs,len(T),*z0.shape[1:])
        xvs = torch.stack([xps[:,:,0],self.Minv.double()@xps[:,:,1]],dim=2)
        return xvs.to(z0.device)

class ChainPendulum(RigidBody):
    def __init__(self,links=2,beams=False,m=1,l=1):
        self.body_graph = nx.Graph()
        if beams:
            self.body_graph.add_node(0,m=m,tether=torch.zeros(2),l=l) #TODO: massful tether
            for i in range(1,links):
                self.body_graph.add_node(i)
                self.body_graph.add_edge(i-1,i,m=m,I=1/12,l=l)
        else:
            self.body_graph.add_node(0,m=m,tether=torch.zeros(2),l=l)
            for i in range(1,links):
                self.body_graph.add_node(i,m=m)
                self.body_graph.add_edge(i-1,i,l=l)


    def body2globalCoords(self,angles_omega):
        d=2; n = len(self.body_graph.nodes); N = angles_omega.shape[0]
        pvs = torch.zeros(N,2,n,d)
        global_position_velocity = torch.zeros(N,2,d)
        length  = self.body_graph.nodes[0]['l']
        global_position_velocity[:,0,:] = self.body_graph.nodes[0]['tether'][None]
        global_position_velocity += self.joint2cartesian(length,angles_omega[...,0])
        pvs[:,:,0] = global_position_velocity
        for (_,j), length in nx.get_edge_attributes(self.body_graph,'l').items():
            global_position_velocity += self.joint2cartesian(length,angles_omega[...,j])
            pvs[:,:,j] = global_position_velocity
        return pvs
    def joint2cartesian(self,length,angle_omega):
        position_vel = torch.zeros(angle_omega.shape[0],2,2)
        position_vel[:,0,0] =  length*angle_omega[:,0].sin()
        position_vel[:,1,0] =  length*angle_omega[:,0].cos()*angle_omega[:,1]
        position_vel[:,0,1] =  -length*angle_omega[:,0].cos()
        position_vel[:,1,1] =  length*angle_omega[:,0].sin()*angle_omega[:,1]
        return position_vel
    def cartesian2angle(self,length,rel_pos_vel):
        raise NotImplementedError
    def global2bodyCoords(self):
        raise NotImplementedError
    def sample_initial_conditions(self,N):
        n = len(self.body_graph.nodes)
        angles_and_angvel = torch.randn(N,2,n) #(N,2,n)
        return self.body2globalCoords(angles_and_angvel)

    def potential(self,x):
        """ Gravity potential """
        return (self.M@x)[...,1].sum(1)

def jvp(y,x,v):
    with torch.enable_grad():
        Jv = torch.autograd.grad(y,[x],[v])[0]
    return Jv
def vjp(y,x,v):
    # Following the trick from https://j-towns.github.io/2017/06/12/A-new-trick.html
    with torch.enable_grad():
        u = torch.ones_like(y,requires_grad=True) # Dummy variable (could take any value)
        Ju = torch.autograd.grad(y,[x],[u],create_graph=True)[0]
        vJ = torch.autograd.grad(Ju,[u],[v])[0]
    return vJ
import numpy as np
def MLE(xt,ts,F,v0=None):
    """ Computes the Maximal Lyapunov exponent using the Power iteration.
        inputs: trajectory [xt (T,*)] dynamics [F] """
    v = torch.randn_like(xt[0]) if v0 is None else v0
    dt = ts[1]-ts[0]
    exps = []
    for i,x in enumerate(xt):
        #for j in range(5):
        x = torch.zeros_like(x, requires_grad=True) + x.detach()
        y = F(ts[i],x[None])[0]
        u = v+vjp(y,x,v).detach()*dt
        #u  = v+ dt*(F(ts[i],(x+1e-7*v)[None])[0]-F(ts[i],x[None])[0])/(1e-7)
        r = (u**2).sum().sqrt().detach()
        v = u/r#P((u/r)[None,:,None])[0,:,0]
        exps += [r.log().item()/dt]#(1/i)*(r.log() - exp)
        #print(r.log()/(100/5000))
    return np.array(exps)#,u

class LyapunovDynamics(nn.Module):
    def __init__(self,F):
        super().__init__()
        self.F = F
    def forward(self,t,xqr):
        n = (xqr.shape[-1]-1)//2
        with torch.enable_grad():
            x = xqr[...,:n] + torch.zeros_like(xqr[...,:n],requires_grad=True)
            q = xqr[...,n:2*n]
            xdot = self.F(t,x)
            DFq = vjp(xdot,x,q)
        qDFq = (q*DFq).sum(-1,keepdims=True)
        qdot = DFq - q*qDFq
        lrdot = qDFq
        xqrdot = torch.cat([xdot,qdot,lrdot],dim=-1)
        return xqrdot


def MLE2(x0,F,ts,**kwargs):
    with torch.no_grad():
        LD = LyapunovDynamics(F)
        x0 = x0.reshape(x0.shape[0],-1)
        q0 = torch.randn_like(x0)
        q0 /= (q0**2).sum(-1,keepdims=True).sqrt()
        lr0 = torch.zeros(x0.shape[0],1,dtype=x0.dtype,device=x0.device)
        Lx0 = torch.cat([x0,q0,lr0],dim=-1)
        Lxt = odeint(LD,Lx0,ts,**kwargs)
        maximal_exponent = Lxt#[...,-1]
    return maximal_exponent
    





# Make animation plots look nicer. Why are there leftover points on the trails?
class Animation2d(object):
    def __init__(self, qt,body, ms=None, box_lim=(-2,2,-3, 2)):
        if ms is None: ms = len(qt)*[6]
        self.qt = qt
        self.G = body.body_graph
        self.fig = plt.figure()
        self.ax = self.fig.add_axes([0, 0, 1, 1])#axes(projection='3d')
        self.ax.set_xlim(box_lim[:2])
        self.ax.set_ylim(box_lim[2:])
        self.ax.set_aspect('equal')
        self.traj_lines = sum([self.ax.plot([],[],'-') for particle in self.qt],[])
        tethers = nx.get_node_attributes(self.G,'tether')
        self.beam_lines = sum([self.ax.plot([],[],'-') for _ in range(len(tethers)+len(self.G.edges))],[])
        self.pts = sum([self.ax.plot([],[],'o',ms=ms[i]) for i in range(len(self.qt))],[])
    def init(self):
        for line,pt in zip(self.traj_lines,self.pts):
            line.set_data([], [])
            pt.set_data([], [])
        for line in self.beam_lines:
            line.set_data([],[])
        return self.traj_lines + self.pts+ self.beam_lines
    def update(self,i=0):
        for node_values,line, pt, trajectory in zip(self.G.nodes.values(),self.traj_lines,self.pts,self.qt):
            x,y = trajectory[:,i-50 if i>50 else 0:i+1]
            line.set_data(x,y)
            if 'm' in node_values: pt.set_data(x[-1:], y[-1:])
        beams = [torch.stack([self.qt[k,:,i],self.qt[l,:,i]],dim=1) for (k,l) in self.G.edges] + \
        [torch.stack([loc,self.qt[k,:,i]],dim=1) for \
            k, loc in nx.get_node_attributes(self.G,'tether').items()]
        for beam,line in zip(beams,self.beam_lines):
            line.set_data(*beam)
        #self.fig.clear()
        self.fig.canvas.draw()
        return self.traj_lines+self.pts+self.beam_lines
    def animate(self):
        return animation.FuncAnimation(self.fig,self.update,frames=self.qt.shape[-1],
                                       interval=33,init_func=self.init,blit=True)