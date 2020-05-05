import torch
import networkx as nx
from oil.utils.utils import export
from biases.systems.rigid_body import RigidBody,BodyGraph,project_onto_constraints
from biases.systems.chain_pendulum import PendulumAnimation
from biases.utils import euler2frame,comEuler2bodyX,read_obj,bodyX2comEuler
import numpy as np

@export
class Gyroscope(RigidBody):
    d=3 # Cartesian Embedding dimension
    D=3 #=3 euler coordinate dimension 
    angular_dims = range(3)
    n=4
    def __init__(self, mass=1, l=1):
        self.body_graph =  BodyGraph()
        self.body_graph.add_extended_nd(0,m=mass,moments=(1,1,1/2))
        self.body_graph.add_joint(0,torch.tensor([0.,0.,-1]),pos2=torch.tensor([0.,0.,0.]))
    
    def sample_initial_conditions(self,N):
        comEulers = torch.randn(N,2,6)
        comEulers[:,1,:3]=0
        comEulers[:,0,:3]=torch.tensor([0.,0.,1.])+1*torch.randn(3)
        comEulers[:,0,3:] = 1*torch.randn(3)
        comEulers[:,1,3:] *=.5
        comEulers[:,1,5] = 6
        bodyX = comEuler2bodyX(comEulers)
        return project_onto_constraints(self.body_graph,bodyX)

    def body2globalCoords(self,eulers):
        """ input: (bs,2,3) output: (bs,2,4,3) """
        coms = torch.zeros_like(eulers)
        comEulers = torch.cat([coms,eulers],dim=-1)
        bodyX = comEuler2bodyX(comEulers)
        # need to offset x,v so that joint is stationary
        # pos joint = 
        body_attachment = self.body_graph.nodes[0]['joint'][0]
        ct = torch.cat([1-body_attachment.sum()[None],body_attachment])
        global_coords_attachment_point = (bodyX*ct[:,None]).sum(-2,keepdims=True) #(bs,2,3)
        return bodyX-global_coords_attachment_point

    def global2bodyCoords(self,bodyX):
        """ input: (bs,2,4,3) output: (bs,2,3)"""
        eulers = bodyX2comEuler(bodyX)[...,3:] # unwrap the euler angles
        eulers[:,0,:] = torch.from_numpy(np.unwrap(eulers[:,0,:].numpy(),axis=0))
        return eulers

    def potential(self, x):
        """ Gravity potential """
        return (self.M @ x)[..., 2].sum(1)


