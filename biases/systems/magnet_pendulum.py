import torch
import networkx as nx
from oil.utils.utils import export
from biases.systems.rigid_body import RigidBody,BodyGraph
from biases.systems.chain_pendulum import PendulumAnimation
import numpy as np
from biases.utils import bodyX2comEuler,comEuler2bodyX, frame2euler,euler2frame

@export
class MagnetPendulum(RigidBody):
    d=3
    n=1
    def __init__(self, mass=1, l=1, q=.05, magnets=5):
        self.arg_string = f"m{mass}l{l}q{q}mn{magnets}"
        self.body_graph = BodyGraph()
        self.body_graph.add_node(0, m=mass, tether=torch.zeros(3), l=l)
        self.q = q  # magnetic moment magnitude
        theta = torch.linspace(0, 2 * np.pi, magnets + 1)[:-1]
        self.magnet_positions = torch.stack(
            [0.1 * theta.cos(), 0.1 * theta.sin(), -(1.2) * l * torch.ones_like(theta)],
            dim=-1,
        )
        self.magnet_dipoles = q*torch.stack([0*theta, 0*theta, torch.ones_like(theta)], dim=-1)  # +z direction

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
    # def sample_initial_conditions(self,N):
    #     angles_vel = torch.randn(N,2,2,1)
    #     return self.body2globalCoords(angles_vel)

    def global2bodyCoords(self, global_pos_vel):
        """ input (bs,2,1,3) output (bs,2,dangular=2,1) """
        *bsT2, n, d = global_pos_vel.shape
        basis = torch.randn(*bsT2,3,d)
        basis[:,:,2:] =global_pos_vel
        basis[:,:,2:] /= (basis[:,:1,2:]**2).sum(-1,keepdims=True).sqrt()
        basis[:,:,1] -=  basis[:,:,2]*(basis[:,:,2]*basis[:,:,1]).sum(-1,keepdims=True)/(basis[:,:,2]**2).sum(-1,keepdims=True)
        basis[:,:,1] /= (basis[:,:1,1]**2).sum(-1,keepdims=True).sqrt()
        basis[:,:,0] -=  basis[:,:,2]*(basis[:,:,2]*basis[:,:,0]).sum(-1,keepdims=True)/(basis[:,:,2]**2).sum(-1,keepdims=True)
        basis[:,:,0] -=  basis[:,:,1]*(basis[:,:,1]*basis[:,:,0]).sum(-1,keepdims=True)/(basis[:,:,1]**2).sum(-1,keepdims=True)
        basis[:,:,0] /= (basis[:,:1,0]**2).sum(-1,keepdims=True).sqrt()
        return frame2euler(basis)[:,:,:2].unsqueeze(-1)
        
    def body2globalCoords(self, angles_omega):
        """ input (bs,2,dangular=2,1) output (bs,2,1,3) """
        bs,_,_,_ = angles_omega.shape
        euler_angles = torch.zeros(bs,2,3)
        euler_angles[:,:,:2] = angles_omega.squeeze(-1)
        zhat = euler2frame(euler_angles)[:,:,2]
        return -zhat.unsqueeze(-2) # (bs,2,1,3)

    def potential(self, x):
        """ Gravity potential """
        gpe = (self.M@x)[...,:,2].sum(-1)# (self.M @ x)[..., 2].sum(1)
        ri = self.magnet_positions
        mi = self.magnet_dipoles[None] # (1,magnets,d)
        r0 = x.squeeze(-2) # (bs,1,d) -> (bs,d)
        m0 = (-1*self.q*r0/(r0**2).sum(-1,keepdims=True))[:,None] #(bs,1,d)
        r0i = ri[None]-r0[:,None] # (bs,magnets,d)
        m0dotr0i = (m0*r0i).sum(-1)#(r0i@m0.transpose(-1,-2)).squeeze(-1) # (bs,magnets)
        midotr0i = (mi*r0i).sum(-1)
        m0dotmi = (m0*mi).sum(-1)
        r0inorm2 = (r0i*r0i).sum(-1)
        dipole_energy = ((-3*m0dotr0i*midotr0i-r0inorm2*m0dotmi)/(4*np.pi*r0inorm2**(5/2))).sum(-1) # (bs,)
        return gpe + dipole_energy #(bs,)


    def __str__(self):
        return f"{self.__class__}{self.arg_string}"

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
