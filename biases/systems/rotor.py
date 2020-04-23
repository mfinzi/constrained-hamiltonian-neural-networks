from oil.utils.utils import export
from biases.systems.rigid_body import RigidBody,BodyGraph
from biases.animation import Animation
import numpy as np
from biases.utils import euler2frame,comEuler2bodyX

@export
class Rotor(RigidBody):
    def __init__(self, mass=1, moments=(1, 2, 3)):
        self.body_graph = BodyGraph()
        self.body_graph.add_extended_nd(0,mass,moments,d=3)
    def sample_initial_conditions(self,N):
        comEulers = torch.randn(N,2,6)
        bodyX = comEuler2bodyX(comEulers)
        return bodyX
    def potential(self,x):
        return 0
