import torch
import networkx as nx
from oil.utils.utils import export
from biases.systems.rigid_body import RigidBody,BodyGraph,project_onto_constraints
#from biases.systems.chain_pendulum import PendulumAnimation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from biases.animation import Animation
from biases.utils import euler2frame,comEuler2bodyX,bodyX2comEuler
from biases.utils import read_obj, compute_moments
import numpy as np


@export
class Gyroscope(RigidBody):
    d=3 # Cartesian Embedding dimension
    D=3 #=3 euler coordinate dimension 
    angular_dims = range(3)
    n=4
    dt=0.02
    integration_time = 2
    def __init__(self, mass=.1, obj='gyro'):
        verts,tris =  read_obj(obj+'.obj')
        verts[:,2] -= verts[:,2].min() # set bottom as 0
        _,com,covar = compute_moments(torch.from_numpy(verts[tris]))
        print(torch.diag(torch.diag(covar).sum()*torch.eye(3)-covar),torch.diag(covar))
        self.obj = (verts,tris,com.numpy())
        self.body_graph =  BodyGraph()
        self.body_graph.add_extended_nd(0,m=mass,moments=100*torch.diag(covar))
        self.body_graph.add_joint(0,-com,pos2=torch.tensor([0.,0.,0.]))
    
    def sample_initial_conditions(self,N):
        # comEulers = torch.randn(N,2,6)
        # comEulers[:,1,:3]=0
        # comEulers[:,0,:3]=torch.tensor([0.,0.,1.])+.3*torch.randn(3)
        # comEulers[:,0,3:] = 1*torch.randn(3)
        # comEulers[:,1,3:] *=0#2#.5
        # comEulers[:,1,5] = 5
        # bodyX = comEuler2bodyX(comEulers)
        # try: return project_onto_constraints(self.body_graph,bodyX,tol=1e-5)
        # except OverflowError: return self.sample_initial_conditions(N)
        eulers = (torch.rand(N,2,3)-.5)*3
        #eulers[:,0,1]*=.2
        eulers[:,1,0]*=3
        eulers[:,1,1]*=.2
        eulers[:,1,2] = (torch.randint(2,size=(N,)).float()*2-1)*(torch.randn(N)+7)*1.5
        return self.body2globalCoords(eulers)

    def body2globalCoords(self,eulers):
        """ input: (bs,2,3) output: (bs,2,4,3) """
        coms = torch.zeros_like(eulers)
        comEulers = torch.cat([coms,eulers],dim=-1)
        bodyX = comEuler2bodyX(comEulers)
        # need to offset x,v so that joint is stationary
        # pos joint = 
        body_attachment = self.body_graph.nodes[0]['joint'][0].to(eulers.device,eulers.dtype)
        ct = torch.cat([1-body_attachment.sum()[None],body_attachment])
        global_coords_attachment_point = (bodyX*ct[:,None]).sum(-2,keepdims=True) #(bs,2,3)
        return bodyX-global_coords_attachment_point

    def global2bodyCoords(self,bodyX):
        """ input: (bs,2,4,3) output: (bs,2,3)"""
        eulers = bodyX2comEuler(bodyX)[...,3:] # unwrap the euler angles
        eulers[:,0,:] = torch.from_numpy(np.unwrap(eulers[:,0,:].numpy(),axis=0)).to(bodyX.device,bodyX.dtype)
        # print(eulers[:,0])
        # assert False
        return eulers

    def potential(self, x):
        """ Gravity potential """
        return 9.81*(self.M @ x)[..., 2].sum(1)

    @property
    def animator(self):
        return RigidAnimation

class RigidAnimation(Animation):
    def __init__(self,qt,body):
        super().__init__(qt,body)
        self.objects['lines'] = sum([self.ax.plot([],[],[],"-",c='k') for _ in range(4)],[])
        
    def update(self,i=0):
        x,y,z = self.qt[i,0].T
        self.objects['lines'][0].set_data([0,x],[0,y])
        self.objects['lines'][0].set_3d_properties([0,z])
        for j in range(1,4):
            self.objects['lines'][j].set_data(*self.qt[i,[0,j]].T[:2])
            self.objects['lines'][j].set_3d_properties(self.qt[i,[0,j]].T[2])
        return super().update(i)
class RigidAnimation2(Animation):
    def __init__(self, qt, body):
        super().__init__(qt, body)
        self.body = body
        self.G = body.body_graph
        vertices,self.triangles,self.ycom = self.body.obj
        x,y,z = self.vertices = vertices.T #(3,n)
        #self.ax.set_aspect("equal")
        self.objects = {
            'pts':self.ax.plot_trisurf(x, y, self.triangles, z, shade=False, color='k',linewidth=0),
            'center':self.ax.scatter([0],[0],[0],s=100,c='b'),
        }
        xyzmin = self.qt[:,0].min(0)#.min(dim=0)[0].min(dim=0)[0]
        xyzmax = self.qt[:,0].max(0)#.max(dim=0)[0].max(dim=0)[0]
        delta = xyzmax-xyzmin
        lower = xyzmin-.1*delta; upper = xyzmax+.1*delta
        self.ax.set_xlim((min(lower),max(upper)))
        self.ax.set_ylim((min(lower),max(upper)))
        self.ax.set_zlim((min(lower),max(upper)))
        
    def init(self):
        #x,y,z = self.vertices
        #self.objects['pts'] = self.ax.plot_trisurf(x, z, self.triangles, y, shade=True, color='white')
        return [self.objects['pts']]

    def update(self, i=0):
        T,n,d = self.qt.shape
        for j in range(1):
            xyz = self.qt[i,:,:]#.data.numpy() #(T,4,3)
            xcom = xyz[0] #(3)
            R = (xyz[1:]-xcom[None,:]) # (3,3)
            new_vertices = R.T@(self.vertices.reshape(*xcom.shape,-1)-self.ycom[:,None])+xcom[:,None]
            self.objects['pts'].set_verts([new_vertices.T])
            #self.objects['pts'].remove()
            #self.objects['pts'] = self.ax.plot_trisurf(x, z, self.triangles, y, shade=True, color='white')
            #if d==3: self.objects['pts'][j].set_3d_properties(xyz[-1:,...,2].T.data.numpy())
        #self.fig.canvas.draw()
        return [self.objects['pts']]

    def animate(self):
        return animation.FuncAnimation(self.fig,self.update,frames=self.qt.shape[0],
                    interval=33,init_func=self.init,blit=True,).to_html5_video()
