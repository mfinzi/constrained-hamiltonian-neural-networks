import torch
import networkx as nx
from oil.utils.utils import export
from biases.systems.rigid_body import RigidBody,BodyGraph,project_onto_constraints
from biases.systems.chain_pendulum import PendulumAnimation
from biases.utils import euler2frame,comEuler2bodyX,read_obj
import numpy as np

@export
class Satellite(RigidBody):
    d=3 # Cartesian Embedding dimension
    D=9 #=3 euler coordinate dimension 
    #angular_dims = range(3)
    n=12
    def __init__(self, mass=1, l=1):
        self.body_graph =  BodyGraph()
        self.body_graph.add_extended_nd(0,m=12,moments=(1/2,1/2,1/2)) #main body
        self.body_graph.add_extended_nd(1,m=3,moments=(1,1,1/8))
        self.body_graph.add_joint(0,torch.tensor([1.,0,0]),1,torch.tensor([0.,0,0]),
            rotation_axis=(torch.tensor([1.,0,0]),torch.tensor([0,0,1.])))
        self.body_graph.add_extended_nd(2,m=3,moments=(1,1,1/8))
        self.body_graph.add_joint(0,torch.tensor([0.,-1,0]),2,torch.tensor([0.,0,0]),
            rotation_axis=(torch.tensor([0.,-1,0]),torch.tensor([0,0,1.])))
        # self.body_graph.add_extended_nd(3,m=3,moments=(1,1,1/8))
        # self.body_graph.add_joint(2,torch.tensor([1.,1,1]),3,torch.tensor([0.,0,0]),
        #     rotation_axis=(torch.tensor([1.,1,1])/np.sqrt(3),torch.tensor([0,0,1.])))
    
    def sample_initial_conditions(self,N):
        bodyX = torch.randn(N,2,self.n,self.d)
        return project_onto_constraints(self.body_graph,bodyX)

    def potential(self, x):
        """ Gravity potential """
        return 0