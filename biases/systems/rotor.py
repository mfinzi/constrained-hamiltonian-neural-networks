from oil.utils.utils import export
from biases.systems.rigid_body import RigidBody, BodyGraph


@export
class Rotor(RigidBody):
    def __init__(self, mass=1, moments=(1, 2, 3)):
        self.body_graph = BodyGraph()
        self.body_graph.add_extended_nd(key=0, m=mass, moments=moments, d=3)

    def sample_initial_conditions(self, N):
        raise NotImplementedError
        # i = self.body.key2id[node]

    def potential(self, x):
        return 0
