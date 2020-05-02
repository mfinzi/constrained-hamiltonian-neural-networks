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
from biases.dynamics.lagrangian import ConstrainedLagrangianDynamics
from biases.systems.rigid_body import rigid_DPhi
from typing import Optional, Tuple, Union
from lie_conv.utils import export, Named
import networkx as nx
import torch.nn.functional as F

@export
class CL(nn.Module, metaclass=Named):  # abstract constrained Hamiltonian network class
    def __init__(self,G,
        dof_ndim: Optional[int] = None,
        angular_dims: Union[Tuple, bool] = tuple(),
        wgrad=True, **kwargs):

        super().__init__(**kwargs)
        if angular_dims != tuple(): print("CH ignores angular_dims")
        self.G = G
        self.nfe = 0
        self.wgrad = wgrad
        self.n_dof = len(G.nodes)
        self.d = self.dof_ndim = dof_ndim
        shape = (self.n_dof,self.d)
        self.dynamics = ConstrainedLagrangianDynamics(self.V, self.Minv,self.DPhi,shape, wgrad=self.wgrad)
        print("CH currently assumes potential energy depends only on q")
        print("CH currently assumes time independent Hamiltonian")
        print("CH assumes positions q are in Cartesian coordinates")
        self.d_moments = nn.ParameterDict(
            {str(d):torch.nn.Parameter(.1*torch.randn(len(d_objs)//(d+1),d+1))
                for d,d_objs in G.d2ids.items()})
        # Mass and 2nd moments of mass for each extended object
    
    def Minv(self,p):
        """ assumes p shape (*,n,a) and n is organized, all the same dimension for now"""
        assert len(self.d_moments)==1, "For now only supporting 1 dimension at a time"
        d = int(list(self.d_moments.keys())[0])

        *start,n,a = p.shape
        N = n//(d+1) # number of extended bodies
        p_reshaped = p.reshape(*start,N,d+1,a) # (*, # separate bodies, # internal body nodes, a)
        inv_moments = torch.exp(self.d_moments[str(d)])
        inv_masses = inv_moments[:,:1] # (N,1)
        padded_inertias_inv = torch.cat([0*inv_masses,inv_moments[:,1:]],dim=-1) # (N,d+1)
        inverse_massed_p = p_reshaped.sum(-2,keepdims=True)*inv_masses[:,:,None]
        total = inverse_massed_p + p_reshaped*padded_inertias_inv[:,:,None]
        return total.reshape(*p.shape)
    

    def DPhi(self, x,v):
        return rigid_DPhi(self.G, x,v)

    def forward(self, t, z):
        self.nfe += 1
        return self.dynamics(t, z)

    def V(self, x):
        """ (bs,nd)"""
        raise NotImplementedError

    def integrate(self, z0, ts, tol=1e-4):
        """ Integrates an initial state forward in time according to the learned Hamiltonian dynamics

        Assumes that z0 = [x0, xdot0] where x0 is in Cartesian coordinates

        Args:
            z0: (bs x 2 x nd) sized
                Tensor representing initial state. N is the batch size
            ts: a length T Tensor representing the time points to evaluate at
            tol: integrator tolerance

        Returns: a bs,T,2,n,d sized Tensor
        """
        bs = z0.shape[0]
        self.nfe = 0
        zt = odeint(self, z0.reshape(bs,-1), ts, rtol=tol, method="rk4").permute(1, 0, 2)
        return zt.reshape(bs, len(ts), 2, self.n_dof, self.dof_ndim)
        


@export
class CLNN(CL):
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

    def V(self, x):
        """ Input is a canonical position variable and the system parameters,
        Args:
            x: (N x n_dof x dof_ndim) sized Tensor representing the position in
            Cartesian coordinates
        Returns: a length N Tensor representing the potential energy
        """
        assert x.ndim == 3
        return self.potential_net(x.reshape(x.size(0), -1))



@export
class CLLC(CL, LieResNet):
    def __init__(self,G,
        dof_ndim: Optional[int] = None,
        angular_dims: Union[Tuple, bool] = tuple(),
        hidden_size=200,num_layers=3,wgrad=True,bn=False,
        group=None,knn=False,nbhd=100,mean=True,**kwargs):
        n_dof = len(G.nodes())
        super().__init__(G=G,dof_ndim=dof_ndim,angular_dims=angular_dims,wgrad=wgrad,
            chin=n_dof,ds_frac=1,num_layers=num_layers,nbhd=nbhd,mean=mean,bn=bn,xyz_dim=dof_ndim,
            group=group or Trivial(dof_ndim),fill=1.0,k=hidden_size,num_outputs=1,cache=False,knn=knn,**kwargs)

    def V(self, x):
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
        return super(CL, self).forward((x, features, mask)).squeeze(-1)
