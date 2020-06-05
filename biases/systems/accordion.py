# import torch
# import networkx as nx
# import numpy as np
# from oil.utils.utils import export
# from biases.systems.rigid_body import RigidBody, BodyGraph, project_onto_constraints
# from biases.animation import Animation
# from biases.systems.chain_pendulum import ChainPendulum
# from biases.systems.magnet_pendulum import MagnetPendulum
# from biases.utils import bodyX2comEuler,comEuler2bodyX, frame2euler,euler2frame
# import copy


# @export
# class ChainPendulum(RigidBody):
#     d=2
#     def __init__(self, links=2, beams=False, m=1, l=1):
#         self.body_graph = BodyGraph()#nx.Graph()
#         self.arg_string = f"n{links}{'b' if beams else ''}m{m}l{l}"
#         assert not beams, "beams temporarily not supported"
#         ms = [.6+.8*np.random.rand() for _ in range(links)] if m is None else links*[m]
#         ls = (links-1)*[l]
#         self.body_graph.add_extended_nd(0, m=ms.pop(), d=0)
#         for i in range(1, links):
#             self.body_graph.add_extended_nd(i, m=ms.pop(), d=0)
#             self.body_graph.add_edge(i - 1, i, l=ls.pop())
#         self.D =self.n = links
#         self.angular_dims = range(links)

#     def potential(self, x):
#         """ Gravity potential """
#         g = 9.81
#         gpe = g*(self.M @ x)[..., 1].sum(1)
#         l0s = ((self.locs[1:]-self.locs[:-1])**2).sum(-1).sqrt().to(x.device,x.dtype)
#         xdist = ((x[:,1:,:]-x[:,:-1,:])**2).sum(-1).sqrt()
#         spring_energy = (.5*self.ks.to(x.device,x.dtype)*(xdist-l0s)**2).sum(1)
#         return gpe+spring_energy
#         return 