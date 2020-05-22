import torch
import networkx as nx
import numpy as np
from oil.utils.utils import export,FixedNumpySeed
from biases.systems.rigid_body import RigidBody, BodyGraph, project_onto_constraints
from biases.animation import Animation
from biases.systems.chain_pendulum import PendulumAnimation
from biases.systems.magnet_pendulum import MagnetPendulum
from biases.utils import bodyX2comEuler,comEuler2bodyX, frame2euler,euler2frame
import copy

@export
class CoupledPendulum(MagnetPendulum):
    d=3
    def __init__(self, bobs=2, m=1, l=1,k=10):
        self.body_graph = BodyGraph()#nx.Graph()
        self.arg_string = f"n{bobs}m{m or 'r'}l{l}"
        with FixedNumpySeed(0):
            ms = [.6+.8*np.random.rand() for _ in range(bobs)] if m is None else bobs*[m]
        ls = bobs*[l]
        self.ks = torch.tensor((bobs-1)*[k]).float()
        self.locs = torch.zeros(bobs,3)
        self.locs[:,0] = 1*torch.arange(bobs).float()
        for i in range(bobs):
            self.body_graph.add_extended_nd(i, m=ms.pop(), d=0,tether=(self.locs[i],ls.pop()))
        self.n = bobs
        self.D = 2*self.n # Spherical coordinates, phi, theta per bob
        self.angular_dims = range(self.D)

    def sample_initial_conditions(self, bs):
        n = len(self.body_graph.nodes)
        angles_and_angvel = .3*torch.randn(bs, 2, 2*n)  # (bs,2,n)
        angles_and_angvel[:,0,1::2] += np.pi/2
        angles_and_angvel[:,0,::2] += np.pi
        z = self.body2globalCoords(angles_and_angvel) #(bs,2,n,d)
        #z[:,0] += self.locs.to(z.device,z.dtype)
        #z[:,0] += .2*torch.randn(bs,n,3)
        #z[:,1,-1] = 1.0*torch.randn(bs,3)
        #z[:,1] = .5*z[:,1] + .4*torch.randn(bs,n,3)
        try: return project_onto_constraints(self.body_graph,z,tol=1e-5)
        except OverflowError: return self.sample_initial_conditions(bs)
    
    # def sample_initial_conditions(self, bs):
    #     n = len(self.body_graph.nodes)
    #     angles_and_angvel = .5*torch.randn(bs, 2, 2*n)  # (bs,2,n)
    #     angles_and_angvel[:,0,:] += np.pi/2
    #     angles_and_angvel[:,0,::2] -= np.pi
    #     z = self.body2globalCoords(angles_and_angvel) #(bs,2,n,d)
    #     #z[:,0] += self.locs.to(z.device,z.dtype)
    #     #z[:,0] += .2*torch.randn(bs,n,3)
    #     z[:,1,-1] = 2*torch.randn(bs,3)
    #     #z[:,1] = .5*z[:,1] + .4*torch.randn(bs,n,3)
    #     try: return project_onto_constraints(self.body_graph,z,tol=1e-5)
    #     except OverflowError: return self.sample_initial_conditions(bs)
    def global2bodyCoords(self, global_pos_vel):
        """ input (bs,2,n,3) output (bs,2,dangular=2n) """
        xyz = copy.deepcopy(global_pos_vel)
        xyz[:,0] -= self.locs.to(xyz.device,xyz.dtype)
        return super().global2bodyCoords(xyz)
    def body2globalCoords(self, angles_omega):
        """ input (bs,2,dangular=2n) output (bs,2,n,3) """
        xyz = super().body2globalCoords(angles_omega)
        xyz[:,0]+=self.locs.to(xyz.device,xyz.dtype)
        return xyz # (bs,2,n,3)

    def potential(self, x):
        """inputs [x (bs,n,d)] Gravity potential
           outputs [V (bs,)] """
        gpe = 9.81*(self.M @ x)[..., 2].sum(1)
        l0s = ((self.locs[1:]-self.locs[:-1])**2).sum(-1).sqrt().to(x.device,x.dtype)
        xdist = ((x[:,1:,:]-x[:,:-1,:])**2).sum(-1).sqrt()
        spring_energy = (.5*self.ks.to(x.device,x.dtype)*(xdist-l0s)**2).sum(1)
        return gpe+spring_energy

    @property
    def animator(self):
        return CoupledPendulumAnimation

def helix(Ns=1000,radius=.05,turns=25):
    t = np.linspace(0,1,Ns)
    xyz = np.zeros((Ns,3))
    xyz[:,0] = np.cos(2*np.pi*Ns*t*turns)*radius
    xyz[:,1] = np.sin(2*np.pi*Ns*t*turns)*radius
    xyz[:,2] = t
    xyz[:,:2][(t>.9)|(t<.1)]=0
    return xyz

def align2ref(refs,vecs):
    """ inputs [refs (n,3), vecs (N,3)]
        outputs [aligned (n,N,3)]
    assumes vecs are pointing along z axis"""
    n,_ = refs.shape
    N,_ = vecs.shape
    norm = np.sqrt((refs**2).sum(-1))
    v = refs/norm[:,None]
    A = np.zeros((n,3,3))
    A[:,:,2] += v
    A[:,2,:] -= v
    M = (np.eye(3)+A+(A@A)/(1+v[:,2,None,None]))*norm[:,None,None]
    return (M[:,None]@vecs[None,...,None]).squeeze(-1)




class CoupledPendulumAnimation(PendulumAnimation):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        empty = self.qt.shape[-1]*[[]]
        self.objects["springs"] = self.ax.plot(*empty,c='k',lw=.6)#sum([self.ax.plot(*empty,c='k',lw=2) for _ in range(self.n-1)],[])
        self.helix = helix(200,turns=10)
    def update(self,i=0):
        diffs = (self.qt[i,1:]-self.qt[i,:-1]).numpy()
        x,y,z = (align2ref(diffs,self.helix)+self.qt[i,:-1][:,None].numpy()).reshape(-1,3).T
        self.objects['springs'][0].set_data(x,y)
        self.objects['springs'][0].set_3d_properties(z)
        return super().update(i)
    # def plot_spring(x, y, theta, L):
    # """Plot the spring from (0,0) to (x,y) as the projection of a helix."""
    # # Spring turn radius, number of turns
    # rs, ns = 0.05, 25
    # # Number of data points for the helix
    # Ns = 1000
    # # We don't draw coils all the way to the end of the pendulum:
    # # pad a bit from the anchor and from the bob by these number of points
    # ipad1, ipad2 = 100, 150
    # w = np.linspace(0, L, Ns)
    # # Set up the helix along the x-axis ...
    # xp = np.zeros(Ns)
    # xp[ipad1:-ipad2] = rs * np.sin(2*np.pi * ns * w[ipad1:-ipad2] / L)
    # # ... then rotate it to align with  the pendulum and plot.
    # R = np.array([[np.cos(theta), -np.sin(theta)],
    #               [np.sin(theta), np.cos(theta)]])
    # xs, ys = - R @ np.vstack((xp, w))
    # ax.plot(xs, ys, c='k', lw=2)