from oil.utils.utils import export
from biases.systems.rigid_body import RigidBody,BodyGraph,project_onto_constraints
from biases.animation import Animation
import numpy as np
from biases.utils import euler2frame,comEuler2bodyX,read_obj
from biases.utils import comEuler2bodyX, bodyX2comEuler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from biases.utils import read_obj, compute_moments
import torch


@export
class Rotor(RigidBody):
    d=3 # Cartesian Embedding dimension
    D=6 #=3 euler + 3 com Total body coordinate dimension 
    angular_dims = range(3,6)#slice(3,None)
    n=4
    dt=0.05
    integration_time=5
    def __init__(self, mass=.1, obj='rotor'):#,moments=(1,2,3)):
        verts,tris =  read_obj(obj+'.obj')
        
        _,com,covar = compute_moments(torch.from_numpy(verts[tris]))
        verts -= com.numpy()[None,:] # set com as 0
        # verts*=100
        eigs,Q = np.linalg.eigh(covar.numpy())
        #print(compute_moments(torch.from_numpy((verts@Q)[tris])))
        moments = torch.diag(covar)
        self.obj = (verts,tris)
        self.body_graph = BodyGraph()
        self.body_graph.add_extended_nd(0,mass,moments,d=3)
    def sample_initial_conditions(self,N):
        comEulers = (2*torch.randn(N,2,6)).clamp(max=3,min=-3)
        comEulers[:,:,:3]*=.1
        bodyX = comEuler2bodyX(comEulers)
        #bodyX = torch.randn(N,2,4,3)
        #bodyX = project_onto_constraints(self.body_graph,bodyX)
        return bodyX
        comEulers = (.75*torch.randn(N,2,6)).clamp(max=1.5,min=-1.5)
        comEulers[:,0,3:]*=.05
        comEulers[:,1,3:]*=1
        # comEulers[:,1,5]*=500
        comEulers[:,:,:3]*=.005
        comEulers[:,1,5]*=4
        #comEulers[]
        bodyX = comEuler2bodyX(comEulers)
        #bodyX = torch.randn(N,2,4,3) + torch.randn(N,1,1,1)
        #bodyX = project_onto_constraints(self.body_graph,bodyX)
        return bodyX
        #return 
    def potential(self,x):
        return 0
    def body2globalCoords(self,comEulers):
        """ input: (bs,2,6) output: (bs,2,4,3) """
        return comEuler2bodyX(comEulers)
    def global2bodyCoords(self,bodyX):
        """ input: (bs,2,4,3) output: (bs,2,6)"""
        comEuler = bodyX2comEuler(bodyX)
        #unwrap euler angles for continuous trajectories
        unwrapped_angles = torch.from_numpy(np.unwrap(comEuler[:,0,3:],axis=0))
        comEuler[:,0,3:] = unwrapped_angles.to(bodyX.device,bodyX.dtype)
        return comEuler

    @property
    def animator(self):
        return RigidAnimation

class RigidAnimation(Animation):
    def __init__(self, qt, body):
        super().__init__(qt, body)
        self.body = body
        self.G = body.body_graph
        vertices,self.triangles = self.body.obj
        x,y,z = self.vertices = vertices.T #(3,n)
        #self.ax.set_aspect("equal")
        self.objects = {
            'pts':self.ax.plot_trisurf(x, y, self.triangles, z, shade=False, color='k'),
            'center':self.ax.scatter([0],[0],[0],s=100,c='b'),
        }
        xyzmin = self.qt[:,0].min(0)#.min(dim=0)[0].min(dim=0)[0]
        xyzmax = self.qt[:,0].max(0)#.max(dim=0)[0].max(dim=0)[0]
        delta = xyzmax-xyzmin
        lower = xyzmin-.1*delta; upper = xyzmax+.1*delta
        self.ax.set_xlim((min(lower),max(upper)))
        self.ax.set_ylim((min(lower),max(upper)))
        self.ax.set_zlim((min(lower),max(upper)))
        # self.ax.set_xlim((-.2,.2))
        # self.ax.set_ylim((-.2,2))
        # self.ax.set_zlim((-.2,.2))
        
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
            new_vertices = R.T@(self.vertices.reshape(*xcom.shape,-1))+xcom[:,None]
            self.objects['pts'].set_verts([new_vertices.T])
            #self.objects['pts'].remove()
            #self.objects['pts'] = self.ax.plot_trisurf(x, z, self.triangles, y, shade=True, color='white')
            #if d==3: self.objects['pts'][j].set_3d_properties(xyz[-1:,...,2].T.data.numpy())
        #self.fig.canvas.draw()
        return [self.objects['pts']]

    def animate(self):
        return animation.FuncAnimation(self.fig,self.update,frames=self.qt.shape[0],
                    interval=33,init_func=self.init,blit=True,).to_html5_video()