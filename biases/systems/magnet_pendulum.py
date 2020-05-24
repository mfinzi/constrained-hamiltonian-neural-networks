import torch
import networkx as nx
from oil.utils.utils import export,FixedNumpySeed
from biases.systems.rigid_body import RigidBody,BodyGraph
from biases.systems.chain_pendulum import PendulumAnimation
import numpy as np
from biases.utils import bodyX2comEuler,comEuler2bodyX, frame2euler,euler2frame

@export
class MagnetPendulum(RigidBody):
    d=3
    n=1
    D = 2
    angular_dims = range(2)
    dt=0.05
    integration_time = 5.
    def __init__(self, mass=3, l=1, q=.3, magnets=2):
        with FixedNumpySeed(0):
            mass = np.random.rand()*.8+2.4 if mass is None else mass
        self.arg_string = f"m{mass or 'r'}l{l}q{q}mn{magnets}"
        self.body_graph = BodyGraph()
        self.body_graph.add_extended_nd(0, m=mass, d=0, tether=(torch.zeros(3),l))
        self.q = q  # magnetic moment magnitude
        theta = torch.linspace(0, 2 * np.pi, magnets + 1)[:-1]
        self.magnet_positions = torch.stack(
            [0.1 * theta.cos(), 0.1 * theta.sin(), -(1.05) * l * torch.ones_like(theta)],
            dim=-1,
        )
        self.magnet_dipoles = q*torch.stack([0*theta, 0*theta, torch.ones_like(theta)], dim=-1)  # +z direction
        # self.magnet_positions = torch.tensor([0.,0., -1.1*l])[None]
        # self.magnet_dipoles = q*torch.tensor([0.,0.,1.])[None]
    def sample_initial_conditions(self, N):
        # phi =torch.rand(N)*2*np.pi
        # phid = .1*torch.randn(N)
        # theta = (4/5)*np.pi + .1*torch.randn(N)
        # thetad = 0.00*torch.randn(N)
        angles_omega = torch.zeros(N,2,2)
        angles_omega[:,0,0] = np.pi+.3*torch.randn(N)
        angles_omega[:,1,0] = .05*torch.randn(N)
        angles_omega[:,0,1] = np.pi/2 + .2*torch.randn(N)
        angles_omega[:,1,1] = .4*torch.randn(N)
        xv = self.body2globalCoords(angles_omega)
        return xv
    # def sample_initial_conditions(self,N):
    #     angles_vel = torch.randn(N,2,2,1)
    #     return self.body2globalCoords(angles_vel)

    def global2bodyCoords(self, global_pos_vel):
        """ input (bs,2,1,3) output (bs,2,dangular=2n) """
        bsT,_ , n, d = global_pos_vel.shape
        x,y,z = global_pos_vel[:,0,:,:].permute(2,0,1)
        xd,yd,zd = global_pos_vel[:,1,:,:].permute(2,0,1)
        x,z,xd,zd = z,-x,zd,-xd # Rotate coordinate system by 90 about y
        phi = torch.atan2(y,x)
        rz = (x**2+y**2).sqrt()
        r = (rz**2 + z**2).sqrt()
        theta = torch.atan2(rz,z)
        phid = (x*yd-y*xd)/rz**2
        thetad = ((xd*x*z+yd*y*z)/rz - rz*zd)/r**2
        angles = torch.stack([phi,theta],dim=-1)
        angles = torch.from_numpy(np.unwrap(angles.numpy(),axis=0)).to(r.device,r.dtype)
        anglesd = torch.stack([phid,thetad],dim=-1)
        angles_omega = torch.stack([angles,anglesd],dim=1)
        return angles_omega.reshape(bsT,2,2*n)

        
    def body2globalCoords(self, angles_omega):
        """ input (bs,2,dangular=2) output (bs,2,1,3) """
        bs,_,n2 = angles_omega.shape
        n = n2//2
        euler_angles = torch.zeros(n*bs,2,3,device=angles_omega.device,dtype=angles_omega.dtype)
        euler_angles[:,:,:2] = angles_omega.reshape(bs,2,n,2).permute(2,0,1,3).reshape(n*bs,2,2)
        # To treat z axis of ZXZ euler angles as spherical coordinates
        # simply set (alpha,beta,gamma) = (phi+pi/2,theta,0)
        euler_angles[:,0,0] += np.pi/2 
        zhat_p = euler2frame(euler_angles)[:,:,2]
        #zhat_p = -euler2frame(euler_angles)[:,:,0]
        zhat_p[:,:,[0,2]]=zhat_p[:,:,[2,0]]
        zhat_p[:,:,0]*=-1 # rotate coordinates by -90 about y
        return zhat_p.reshape(n,bs,2,3).permute(1,2,0,3) # (bs,2,n,3)

    def potential(self, x):
        """ Gravity potential """
        gpe = 9.81*(self.M@x)[...,:,2].sum(-1)# (self.M @ x)[..., 2].sum(1)
        ri = self.magnet_positions.to(x.device,x.dtype)
        mi = self.magnet_dipoles[None].to(x.device,x.dtype) # (1,magnets,d)
        r0 = x.squeeze(-2) # (bs,1,d) -> (bs,d)
        m0 = (self.q*r0/(r0**2).sum(-1,keepdims=True))[:,None] #(bs,1,d)
        r0i = (ri[None]-r0[:,None]) # (bs,magnets,d)
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
                self.body.magnet_positions[:, 2].cpu().data.numpy()
            )
        self.ax.set_xlim((-1, 1))
        self.ax.set_ylim((-1, 1))
        if d == 3:
            self.ax.set_zlim((-1.3, 0))
        self.ax.view_init(azim=0,elev=80)#0, 0)  # azim=0,elev=80)#0,30)#azim=0,elev=80)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        # self.magnets.set
        #  self.objects = {
        #     'pts':sum([self.ax.plot(*empty, "o", ms=6) for i in range(n)], []),
        #     'traj_lines':sum([self.ax.plot(*empty, "-") for _ in range(n)], []),
        # }
