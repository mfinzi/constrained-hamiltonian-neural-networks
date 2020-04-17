import torch
import networkx as nx
from oil.utils.utils import export
from biases.systems.rigid_body import RigidBody,BodyGraph
from biases.animation import Animation
import numpy as np

@export
class Rotor(RigidBody):
    def __init__(self, mass=1,moments=(1,2,3)):
        self.body_graph = BodyGraph()
        self.body_graph.add_extended_nd(0,m=mass,moments,d=3):
    def sample_initial_conditions(self,N):
        raise NotImplementedError
        #i = self.body.key2id[node]
    def potential(self,x):
        return 0
