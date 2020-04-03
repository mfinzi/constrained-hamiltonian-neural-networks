import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import functools
from oil.utils.utils import Named, export, Expression
from torchdiffeq import odeint_adjoint as odeint
from torch.autograd import grad

class LagrangianDynamics(nn.Module):
    """ Defines the dynamics given a hamiltonian. If wgrad=True, the dynamics can be backproped."""

    def __init__(self, L, wgrad=False):
        super().__init__()
        self.L = L
        self.wgrad = wgrad
        self.nfe = 0

    def forward(self, t, z):
        self.nfe += 1
        bs,twoD = z.shape
        D = twoD//2
        with torch.enable_grad():
            q = z[...,:D]
            v = z[...,D:]
            q = q+ torch.zeros_like(q,requires_grad=True)
            v = v+ torch.zeros_like(v,requires_grad=True)
            z = torch.cat([q,v],dim=-1)
            L = self.L(z).sum()  # elements in mb are independent, gives mb gradients
            dL_dz = grad(L, z, create_graph=True)[0]  # gradient
            dL_dq = dL_dz[...,:D]
            dL_dv = dL_dz[...,D:]
            Fv = -grad((dL_dq*v.detach()).sum(),v,create_graph=True,retain_graph=True)[0] # elements in mb are independent, gives mb gradients
            M = torch.zeros(bs,D,D,device=z.device,dtype=z.dtype)
            I = torch.eye(D,device=z.device,dtype=z.dtype)
            for i in range(D):
                ei = I[i]
                M[:,:,i] = grad((dL_dv*ei).sum(),v,create_graph=True,retain_graph=True)[0]
            F = (dL_dq+Fv).unsqueeze(-1)
            a = torch.solve(F,M)[0].squeeze(-1)
            dynamics = torch.cat([v,a],dim=-1)#+Fv#
        return dynamics

def PendulumLagrangian(z):
    q = z[...,0]; v = z[...,1]
    return v*v/2 + (q.cos()-1)
def LagrangianFlow(L,z0,T,higher=False):
    dynamics = lambda t,z: LagrangianDynamics(L,higher)(t,z)
    return odeint(dynamics, z0, T,rtol=1e-6).permute(1,0,2)