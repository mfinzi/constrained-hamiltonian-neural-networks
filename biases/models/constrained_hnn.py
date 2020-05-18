import torch
import torch.nn as nn
from torchdiffeq import odeint
from lie_conv.lieConv import LieResNet
from lie_conv.lieGroups import Trivial
from biases.models.utils import FCtanh, Linear, Reshape
from biases.dynamics.hamiltonian import (
    EuclideanT,
    ConstrainedHamiltonianDynamics,
)
from biases.systems.rigid_body import rigid_DPhi
from typing import Optional, Tuple, Union
from lie_conv.utils import export, Named
import networkx as nx
import torch.nn.functional as F

@export
class CH(nn.Module, metaclass=Named):  # abstract constrained Hamiltonian network class
    def __init__(self,G,
        dof_ndim: Optional[int] = None,
        angular_dims: Union[Tuple, bool] = tuple(),
        wgrad=True, **kwargs):

        super().__init__(**kwargs)
        if angular_dims != tuple():
            print("CH ignores angular_dims")
        self.G = G
        self.nfe = 0
        self.wgrad = wgrad
        self.n_dof = len(G.nodes)
        self.dof_ndim = dof_ndim
        self.q_ndim = self.n_dof * self.dof_ndim
        self.dynamics = ConstrainedHamiltonianDynamics(self.H, self.DPhi, wgrad=self.wgrad)
        #self._Minv = torch.nn.Parameter(torch.eye(self.n_dof))
        print("CH currently assumes potential energy depends only on q")
        print("CH currently assumes time independent Hamiltonian")
        print("CH assumes positions q are in Cartesian coordinates")
        #self.moments = torch.nn.Parameter(torch.ones(self.n_dof,self.n_dof))
        #self.masses = torch.nn.Parameter(torch.zeros(self.n_dof))
        #self.moments = torch.nn.Parameter(torch.zeros(self.dof_ndim,self.n_dof))
        self.d_moments = nn.ParameterDict(
            {str(d):torch.nn.Parameter(.1*torch.randn(len(d_objs)//(d+1),d+1)) # N,d+1
                for d,d_objs in G.d2ids.items()})

    def Minv(self,p):
        """ assumes p shape (*,n,a) and n is organized, all the same dimension for now"""
        assert len(self.d_moments)==1, "For now only supporting 1 dimension at a time"
        d = int(list(self.d_moments.keys())[0])

        *start,n,a = p.shape
        N = n//(d+1) # number of extended bodies
        p_reshaped = p.reshape(*start,N,d+1,a) # (*, # separate bodies, # internal body nodes, a)
        inv_moments = torch.exp(-self.d_moments[str(d)])
        inv_masses = inv_moments[:,:1] # (N,1)
        if d==0: return (inv_masses.unsqueeze(-1)*p_reshaped).reshape(*p.shape)# no inertia for point masses
        padded_inertias_inv = torch.cat([0*inv_masses,inv_moments[:,1:]],dim=-1) # (N,d+1)
        inverse_massed_p = p_reshaped.sum(-2,keepdims=True)*inv_masses[:,:,None]
        total = inverse_massed_p + p_reshaped*padded_inertias_inv[:,:,None]
        return total.reshape(*p.shape)

    def M(self,v):
        """ assumes v has shape (*,n,a) and n is organized, all the same dimension for now"""
        assert len(self.d_moments)==1, "For now only supporting 1 dimension at a time"
        d = int(list(self.d_moments.keys())[0])
        *start,n,a = v.shape 
        N = n//(d+1) # number of extended bodies
        v_reshaped = v.reshape(*start,N,d+1,a) # (*, # separate bodies, # internal body nodes, a)       
        moments = torch.exp(self.d_moments[str(d)])
        masses = moments[:,:1]
        if d==0: return (masses.unsqueeze(-1)*v_reshaped).reshape(*v.shape) # no inertia for point masses
        a00 = (masses + moments[:,1:].sum(-1,keepdims=True)).unsqueeze(-1) #(N,1,1)
        ai0 = a0i = -moments[:,1:].unsqueeze(-1) #(N,d,1)
        p0 = a00*v[...,:1,:] + (a0i*v[...,1:,:]).sum(-2,keepdims=True)
        aii = moments[:,1:].unsqueeze(-1) # (N,d,1)
        
        pi = ai0*v[...,:1,:] +aii*v[...,1:,:]
        return torch.cat([p0,pi],dim=-2).reshape(*v.shape)


    def H(self, t, z):
        """ Compute the Hamiltonian H(t, x, p)
        Args:
            t: Scalar Tensor representing time
            z: N x D Tensor of the N different states in D dimensions.
                Assumes that z is [x, p] where x is in Cartesian coordinates.

        Returns: Size N Hamiltonian Tensor
        """
       # assert (t.ndim == 0) and (z.ndim == 2)
        assert z.size(-1) == 2 * self.n_dof * self.dof_ndim

        x, p = z.chunk(2, dim=-1)
        x = x.reshape(-1, self.n_dof, self.dof_ndim)
        p = p.reshape(-1, self.n_dof, self.dof_ndim)

        T = EuclideanT(p, self.Minv)
        V = self.compute_V(x)
        return T + V

    def DPhi(self, zp):
        bs,n,d = zp.shape[0],self.n_dof,self.dof_ndim
        x,p = zp.reshape(bs,2,n,d).unbind(dim=1)
        v = self.Minv(p)
        DPhi = rigid_DPhi(self.G, x, v)
        # Convert d/dv to d/dp
        #DPhi[:,1] = 
        DPhi = torch.cat([DPhi[:,:1],self.Minv(DPhi[:,1].reshape(bs,n,-1)).reshape(DPhi[:,1:].shape)],dim=1)
        return DPhi.reshape(bs,2*n*d,-1)

    def forward(self, t, z):
        self.nfe += 1
        return self.dynamics(t, z)

    def compute_V(self, x):
        raise NotImplementedError

    def integrate(self, z0, ts, tol=1e-4, method="rk4"):
        """ Integrates an initial state forward in time according to the learned Hamiltonian dynamics

        Assumes that z0 = [x0, xdot0] where x0 is in Cartesian coordinates

        Args:
            z0: (N x 2 x n_dof x dof_ndim) sized
                Tensor representing initial state. N is the batch size
            ts: a length T Tensor representing the time points to evaluate at
            tol: integrator tolerance

        Returns: a N x T x 2 x n_dof x d sized Tensor
        """
        assert (z0.ndim == 4) and (ts.ndim == 1)
        assert z0.size(-1) == self.dof_ndim
        assert z0.size(-2) == self.n_dof
        bs = z0.size(0)
        #z0 = z0.reshape(N, -1)  # -> N x (2 * n_dof * dof_ndim) =: N x D
        x0, xdot0 = z0.chunk(2, dim=1)
        p0 = self.M(xdot0)

        self.nfe = 0
        xp0 = torch.stack([x0, p0], dim=1).reshape(bs,-1)
        xpt = odeint(self, xp0, ts, rtol=tol, method=method)
        xpt = xpt.permute(1, 0, 2)  # T x bs x D -> bs x T x D
        xpt = xpt.reshape(bs, len(ts), 2, self.n_dof, self.dof_ndim)
        xt, pt = xpt.chunk(2, dim=-3)
        # TODO: make Minv @ pt faster by L(L^T @ pt)
        vt = self.Minv(pt)  # Minv [n_dof x n_dof]. pt [bs, T, 1, n_dof, dof_ndim]
        xvt = torch.cat([xt, vt], dim=-3)
        return xvt


@export
class CHNN(CH):
    def __init__(self,G,
        dof_ndim: Optional[int] = None,
        angular_dims: Union[Tuple, bool] = tuple(),
        hidden_size: int = 200,
        num_layers=3,
        wgrad=True,
        **kwargs
    ):
        super().__init__(G=G, dof_ndim=dof_ndim, angular_dims=angular_dims, wgrad=wgrad, **kwargs
        )
        n = len(G.nodes())
        chs = [n * self.dof_ndim] + num_layers * [hidden_size]
        self.potential_net = nn.Sequential(
            *[FCtanh(chs[i], chs[i + 1], zero_bias=True, orthogonal_init=True)
                for i in range(num_layers)],
            Linear(chs[-1], 1, zero_bias=True, orthogonal_init=True),
            Reshape(-1)
        )

    def compute_V(self, x):
        """ Input is a canonical position variable and the system parameters,
        Args:
            x: (N x n_dof x dof_ndim) sized Tensor representing the position in
            Cartesian coordinates
        Returns: a length N Tensor representing the potential energy
        """
        assert x.ndim == 3
        return self.potential_net(x.reshape(x.size(0), -1))



@export
class CHLC(CH, LieResNet):
    def __init__(self,G,
        dof_ndim: Optional[int] = None,
        angular_dims: Union[Tuple, bool] = tuple(),
        hidden_size=200,num_layers=3,wgrad=True,bn=False,
        group=None,knn=False,nbhd=100,mean=True,**kwargs):
        n_dof = len(G.nodes())
        super().__init__(G=G,dof_ndim=dof_ndim,angular_dims=angular_dims,wgrad=wgrad,
            chin=n_dof,ds_frac=1,num_layers=num_layers,nbhd=nbhd,mean=mean,bn=bn,xyz_dim=dof_ndim,
            group=group or Trivial(dof_ndim),fill=1.0,k=hidden_size,num_outputs=1,cache=False,knn=knn,**kwargs)

    def compute_V(self, x):
        """ Input is a canonical position variable and the system parameters,
        Args:
            x: (N x n_dof x dof_ndim) sized Tensor representing the position in
            Cartesian coordinates
        Returns: a length N Tensor representing the potential energy
        """
        assert x.ndim == 3
        mask = ~torch.isnan(x[..., 0])
        # features = torch.zeros_like(x[...,:1])
        bs, n, d = x.shape
        features = torch.eye(n, device=x.device, dtype=x.dtype)[None].repeat((bs, 1, 1))
        return super(CH, self).forward((x, features, mask)).squeeze(-1)
