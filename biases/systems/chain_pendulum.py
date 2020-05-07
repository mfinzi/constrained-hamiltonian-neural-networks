import torch
import networkx as nx
import numpy as np
from oil.utils.utils import export
from biases.systems.rigid_body import RigidBody, BodyGraph, project_onto_constraints
from biases.animation import Animation


@export
class ChainPendulum(RigidBody):
    d=2
    def __init__(self, links=2, beams=False, m=None, l=None):
        self.body_graph = BodyGraph()#nx.Graph()
        self.arg_string = f"n{links}{'b' if beams else ''}m{m}l{l}"
        assert not beams, "beams temporarily not supported"
        ms = [.6+.8*np.random.rand() for _ in range(links)] if m is None else links*[m]
        ls = [.6+.8*np.random.rand() for _ in range(links)] if l is None else links*[l]
        self.body_graph.add_extended_nd(0, m=ms.pop(), d=0,tether=(torch.zeros(2),ls.pop()))
        for i in range(1, links):
            self.body_graph.add_extended_nd(i, m=ms.pop(), d=0)
            self.body_graph.add_edge(i - 1, i, l=ls.pop())
        self.D =self.n = links
        self.angular_dims = range(links)

    def body2globalCoords(self, angles_omega):
        d = 2
        n = len(self.body_graph.nodes)
        N = angles_omega.shape[0]
        pvs = torch.zeros(N, 2, n, d,device=angles_omega.device,dtype=angles_omega.dtype)
        global_position_velocity = torch.zeros(N, 2, d,device=angles_omega.device,dtype=angles_omega.dtype)
        length = self.body_graph.nodes[0]["tether"][1]
        global_position_velocity[:, 0, :] = self.body_graph.nodes[0]["tether"][0][None]
        global_position_velocity += self.joint2cartesian(length, angles_omega[..., 0])
        pvs[:, :, 0] = global_position_velocity
        for (_, j), length in nx.get_edge_attributes(self.body_graph, "l").items():
            global_position_velocity += self.joint2cartesian(length, angles_omega[..., j])
            pvs[:, :, j] = global_position_velocity
        return pvs

    def joint2cartesian(self, length, angle_omega):
        position_vel = torch.zeros(angle_omega.shape[0], 2, 2,device=angle_omega.device,dtype=angle_omega.dtype)
        position_vel[:, 0, 0] = length * angle_omega[:, 0].sin()
        position_vel[:, 1, 0] = length * angle_omega[:, 0].cos() * angle_omega[:, 1]
        position_vel[:, 0, 1] = -length * angle_omega[:, 0].cos()
        position_vel[:, 1, 1] = length * angle_omega[:, 0].sin() * angle_omega[:, 1]
        return position_vel

    def cartesian2angle(self, rel_pos_vel):
        x, y = rel_pos_vel[:, 0].T
        vx, vy = rel_pos_vel[:, 1].T
        angle = torch.atan2(x, -y)
        omega = torch.where(angle < 1e-2, vx / (-y), vy / x)
        angle_unwrapped = torch.from_numpy(np.unwrap(angle.numpy(), axis=0)).to(
            x.device, x.dtype
        )
        return torch.stack([angle_unwrapped, omega], dim=1)

    def global2bodyCoords(self, global_pos_vel):
        N, _, n, d = global_pos_vel.shape
        *bsT2, n, d = global_pos_vel.shape
        angles_omega = torch.zeros(
            *bsT2, n, device=global_pos_vel.device, dtype=global_pos_vel.dtype
        )
        start_position_velocity = torch.zeros(*bsT2, d,device=angles_omega.device,dtype=angles_omega.dtype)
        start_position_velocity[..., 0, :] = self.body_graph.nodes[0]["tether"][0][None]
        rel_pos_vel = global_pos_vel[..., 0, :] - start_position_velocity
        angles_omega[..., 0] += self.cartesian2angle(rel_pos_vel)
        start_position_velocity += rel_pos_vel
        for (_, j), length in nx.get_edge_attributes(self.body_graph, "l").items():
            rel_pos_vel = global_pos_vel[..., j, :] - start_position_velocity
            angles_omega[..., j] += self.cartesian2angle(rel_pos_vel)
            start_position_velocity += rel_pos_vel
        return angles_omega

    def sample_initial_conditions(self, N):
        n = len(self.body_graph.nodes)
        angles_and_angvel = torch.randn(N, 2, n)  # (N,2,n)
        z = self.body2globalCoords(angles_and_angvel)
        #z = torch.randn(N,2,n,2)
        z[:,0] += .2*torch.randn(N,n,2)
        z[:,1] = .5*z[:,1] + .4*torch.randn(N,n,2)
        return project_onto_constraints(self.body_graph,z,tol=1e-5)
        
        # return 

    def potential(self, x):
        """ Gravity potential """
        return (self.M @ x)[..., 1].sum(1)

    def __str__(self):
        return f"{self.__class__}{self.arg_string}"

    def __repr__(self):
        return str(self)

    @property
    def animator(self):
        return PendulumAnimation

class ChainPendulumV2(ChainPendulum):
    def __init__(self, links=2, beams=False, m=1, l=1):
        self.body_graph = BodyGraph()#nx.Graph()
        self.arg_string = f"n{links}{'b' if beams else ''}m{m}l{l}"
        beam_moments = torch.tensor([m*l*l/12])
        if beams:
            self.body_graph.add_extended_nd(0, m=m,moments=beam_moments, d=1)
            self.body_graph.add_joint(0,torch.tensor([l/2]))
            for i in range(1, links):
                self.body_graph.add_extended_nd(i, m=m,moments=beam_moments, d=1)
                self.body_graph.add_joint(i-1,torch.tensor([-l/2]),i,torch.tensor([l/2]))
        else:
            self.body_graph.add_node(0, m=m, tether=torch.zeros(2), l=l)
            for i in range(1, links):
                self.body_graph.add_node(i, m=m)
                self.body_graph.add_edge(i - 1, i, l=l)

class PendulumAnimation(Animation):
    def __init__(self, qt, body):
        super().__init__(qt, body)
        self.body = body
        self.G = body.body_graph
        empty = self.qt.shape[-1] * [[]]
        n_beams = len(nx.get_node_attributes(self.G, "tether")) + len(self.G.edges)
        self.objects["beams"] = sum(
            [self.ax.plot(*empty, "-") for _ in range(n_beams)], []
        )

    def update(self, i=0):
        beams = [
            torch.stack([self.qt[i, k, :], self.qt[i, l, :]], dim=1)
            for (k, l) in self.G.edges
        ] + [
            torch.stack(
                [loc.to(self.qt.device, self.qt.dtype), self.qt[i, k, :]], dim=1
            )
            for k, (loc,_) in nx.get_node_attributes(self.G, "tether").items()
        ]
        for beam, line in zip(beams, self.objects["beams"]):
            line.set_data(*beam[:2])
            if self.qt.shape[-1] == 3:
                line.set_3d_properties(beam[2].data.numpy())
        return super().update(i)
