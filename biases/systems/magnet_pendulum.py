import torch
import networkx as nx
from oil.utils.utils import export
from biases.hamiltonian import RigidBody
from biases.chainPendulum import PendulumAnimation
import numpy as np


@export
class MagnetPendulum(RigidBody):
    def __init__(self, mass=1, l=1, q=1, magnets=4):
        self.body_graph = nx.Graph()
        self.body_graph.add_node(0, m=mass, tether=torch.zeros(3), l=l)
        self.q = q  # magnetic moment magnitude
        theta = torch.linspace(0, 2 * np.pi, magnets + 1)[:-1]
        self.magnet_positions = torch.stack(
            [0.1 * theta.cos(), 0.1 * theta.sin(), -(1.2) * l * torch.ones_like(theta)],
            dim=-1,
        )
        self.magnet_dipoles = q * torch.stack(
            [0 * theta, 0 * theta, torch.ones_like(theta)], dim=-1
        )  # +z direction

    def sample_initial_conditions(self, N):
        n = len(self.body_graph.nodes)
        xv = 0.2 * torch.randn(N, 2, 1, 3)  # (N,2,n,d)
        xv[:, 0, :, :] += torch.tensor([0.0, 0.0, -1.0])
        xv[:, 0, :, :] /= (xv[:, 0, :, :] ** 2).sum(-1, keepdims=True).sqrt()

        xv[:, 1, :, :] -= xv[:, 0, :, :] * (xv[:, 0, :, :] * xv[:, 1, :, :]).sum(
            -1, keepdims=True
        )
        # xv[:,1,:,:]*= .1
        return xv

    def potential(self, x):
        """ Gravity potential """
        gpe = x[..., 0, 2]  # (self.M @ x)[..., 2].sum(1)
        ri = self.magnet_positions
        mi = self.magnet_dipoles[None]  # (1,magnets,d)
        r0 = x.unsqueeze(-2)  # (bs,1,d) -> (bs,d)
        m0 = (-1 * self.q * r0 / (r0 ** 2).sum(-1, keepdims=True))[:, None]  # (bs,1,d)
        r0i = ri[None] - r0[:, None]  # (bs,magnets,d)
        m0dotr0i = (m0 * r0i).sum(
            -1
        )  # (r0i@m0.transpose(-1,-2)).squeeze(-1) # (bs,magnets)
        midotr0i = (mi * r0i).sum(-1)
        m0dotmi = (m0 * mi).sum(-1)
        r0inorm2 = (r0i * r0i).sum(-1)
        dipole_energy = (
            3
            * (m0dotr0i * midotr0i - r0inorm2 * m0dotmi)
            / (4 * np.pi * r0inorm2 ** (5 / 2))
        )  # (bs,magnets)
        return gpe - dipole_energy.sum(-1)  # (bs,)

    def __str__(self):
        return f"{self.__class__}{len(self.body_graph.nodes)}"

    def __repr__(self):
        return str(self)

    @property
    def animator(self):
        return MagnetPendulumAnimation


class MagnetPendulumAnimation(PendulumAnimation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d = self.qt.shape[-1]
        empty = d * [[]]
        self.magnets = self.ax.plot(*empty, "o", ms=8)[0]
        self.magnets.set_data(*self.body.magnet_positions[:, :2].T)
        if d == 3:
            self.magnets.set_3d_properties(
                self.body.magnet_positions[:, 2].data.numpy()
            )
        self.ax.set_xlim((-1, 1))
        self.ax.set_ylim((-1, 1))
        if d == 3:
            self.ax.set_zlim((-1.3, 0))
        self.ax.view_init(0, 0)  # azim=0,elev=80)#0,30)#azim=0,elev=80)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        # self.magnets.set
        #  self.objects = {
        #     'pts':sum([self.ax.plot(*empty, "o", ms=6) for i in range(n)], []),
        #     'traj_lines':sum([self.ax.plot(*empty, "-") for _ in range(n)], []),
        # }
