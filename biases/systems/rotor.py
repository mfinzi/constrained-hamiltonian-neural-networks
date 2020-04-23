import torch
import networkx as nx
from oil.utils.utils import export
from biases.systems.rigid_body import RigidBody,BodyGraph
from biases.animation import Animation
import numpy as np
from biases.utils import euler2frame,comEuler2bodyX,read_obj
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

@export
class Rotor(RigidBody):
    d=3
    def __init__(self, mass=1,moments=(1,2,3)):
        self.body_graph = BodyGraph()
        self.body_graph.add_extended_nd(0,mass,moments,d=3)
    def sample_initial_conditions(self,N):
        comEulers = torch.randn(N,2,6)
        bodyX = comEuler2bodyX(comEulers)
        return bodyX
    def potential(self,x):
        return 0

    @property
    def animator(self):
        return RigidAnimation
    def __str__(self):
        return "Rotor"
    def __repr__(self):
        return str(self)

class RigidAnimation(Animation):
    def __init__(self, qt, body):
        super().__init__(qt, body)
        self.body = body
        self.G = body.body_graph
        vertices,self.triangles = read_obj("10540_Tennis_racket_V2_L3.obj")
        x,y,z = self.vertices = .3*vertices.T #(3,n)
        #self.ax.set_aspect("equal")
        self.objects = {
            'pts':self.ax.plot_trisurf(x, z, self.triangles, y, shade=True, color='white')
        }
        
    def init(self):
        #x,y,z = self.vertices
        #self.objects['pts'] = self.ax.plot_trisurf(x, z, self.triangles, y, shade=True, color='white')
        return [self.objects['pts']]
    def update(self, i=0):
        T,n,d = self.qt.shape
        for j in range(1):
            xyz = self.qt[i,:,:].data.numpy() #(T,4,3)
            xcom = xyz[0] #(3)
            R = (xyz[1:]-xcom[None,:]) # (T,3,3)
            new_vertices = (R@self.vertices).reshape(*xcom.shape,-1)+xcom[:,None]
            x,y,z = new_vertices
            #self.objects['pts'].set_verts([new_vertices])
            self.objects['pts'].remove()
            self.objects['pts'] = self.ax.plot_trisurf(x, z, self.triangles, y, shade=True, color='white')
            #if d==3: self.objects['pts'][j].set_3d_properties(xyz[-1:,...,2].T.data.numpy())
        #self.fig.canvas.draw()
        return [self.objects['pts']]

    def animate(self):
        return animation.FuncAnimation(self.fig,self.update,frames=self.qt.shape[0],
                    interval=33,init_func=self.init,blit=True,).to_html5_video()