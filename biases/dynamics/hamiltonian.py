import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, Union
from oil.utils.utils import export


@export
class HamiltonianDynamics(nn.Module):
    """ Defines the dynamics given a Hamiltonian.

    Args:
        H: A callable function that takes in q and p concatenated together and returns H(q, p)
        wgrad: If True, the dynamics can be backproped.
    """

    def __init__(self, H: Callable[[Tensor, Tensor], Tensor], wgrad: bool = True):
        super().__init__()
        self.H = H
        self.wgrad = wgrad
        self.nfe = 0

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        """ Computes a batch of `NxD` time derivatives of the state `z` at time `t`
        Args:
            t: Scalar Tensor of the current time
            z: N x D Tensor of the N different states in D dimensions
        """
        assert (t.ndim == 0) and (z.ndim == 2)
        self.nfe += 1
        with torch.enable_grad():
            z = torch.zeros_like(z, requires_grad=True) + z
            H = self.H(t, z).sum()  # elements in mb are independent, gives mb gradients
            dH = torch.autograd.grad(H, z, create_graph=self.wgrad)[0]  # gradient
            if torch.isnan(dH).any():
                raise RuntimeError("NaNs in dH")
        return J(dH.unsqueeze(-1)).squeeze(-1)


@export
class ConstrainedHamiltonianDynamics(nn.Module):
    """ Defines the Constrained Hamiltonian dynamics given a Hamiltonian and
    gradients of constraints.

    Args:
        H: A callable function that takes in q and p and returns H(q, p)
        DPhi:
        wgrad: If True, the dynamics can be backproped.
    """

    def __init__(
        self,
        H: Callable[[Tensor, Tensor], Tensor],
        DPhi: Callable[[Tensor], Tensor],
        #Phi=None,
        wgrad: bool = True,
    ):
        super().__init__()
        self.H = H
        #self.Phi=Phi
        self.DPhi = DPhi
        self.wgrad = wgrad
        self.nfe = 0

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        """ Computes a batch of `NxD` time derivatives of the state `z` at time `t`
        Args:
            t: Scalar Tensor of the current time
            z: N x D Tensor of the N different states in D dimensions
        """
        assert (t.ndim == 0) and (z.ndim == 2)
        self.nfe += 1
        with torch.enable_grad():
            z = torch.zeros_like(z, requires_grad=True) + z
            P = Proj(self.DPhi(z))
            H = self.H(t, z).sum()  # elements in mb are independent, gives mb gradients
            dH = torch.autograd.grad(H, z, create_graph=self.wgrad)[0]  # gradient
        dynamics = P(J(dH.unsqueeze(-1))).squeeze(-1)
        # if self.Phi is not None:
        #     dynamics += stabilization(self.DPhi(z),self.Phi(z))
        return dynamics


def J(M):
    """ applies the J matrix to another matrix M.
        input: M (*,2nd,b), output: J@M (*,2nd,b)"""
    *star, D, b = M.shape
    JM = torch.cat([M[..., D // 2 :, :], -M[..., : D // 2, :]], dim=-2)
    return JM

def stabilization(DPhi,Phi):
    DPhiT = DPhi.transpose(-1, -2)
    X,_ = torch.solve(Phi.unsqueeze(-1),DPhiT @ J(DPhi))
    return -J(DPhi@X).squeeze(-1)

def Proj(DPhi):
    if DPhi.shape[-1]==0: return lambda M:M # (no constraints)
    def _P(M):
        DPhiT = DPhi.transpose(-1, -2)
        X, _ = torch.solve(DPhiT @ M, DPhiT @ J(DPhi))
        return M - J(DPhi @ X)

    return _P

@export
def EuclideanT(p: Tensor, Minv: Union[Callable[[Tensor], Tensor]]) -> Tensor:
    """p^T Minv p/2 kinetic energy in Euclidean space.

    Note that in Euclidean space, Minv only mixes degrees of freedom, not their individual dimensions

    Args:
        p: bs x ndof x D Tensor representing the canonical momentum in Cartesian coordinates
        Minv: bs x ndof x ndof Tensor representing the inverse mass matrix. Can be a
            callable that computes Minv(p) as well
    """

    assert p.ndim == 3
    Minv_p = Minv(p) if callable(Minv) else Minv.matmul(p)
    assert Minv_p.ndim == 3
    T = (p * Minv_p).sum((-1, -2)) / 2.0
    return T

@export
def GeneralizedT(p: Tensor, Minv: Union[Callable[[Tensor], Tensor]]) -> Tensor:
    """p^T Minv p/2 kinetic energy in generalized coordinates

    Note that in non-Euclidean space, Minv mixes every coordinate

    Args:
        p: bs x D Tensor representing the canonical momentum.
        Minv: bs x D x D Tensor representing the inverse mass matrix. Can be a
            callable that computes Minv(p) as well
    """
    assert p.ndim == 2
    Minv_p = Minv(p) if callable(Minv) else Minv.matmul(p.unsqueeze(-1)).squeeze(-1)
    assert Minv_p.ndim == 2
    T = (p * Minv_p).sum((-1,)) / 2.0
    return T


@export
class ConstrainedLagrangianDynamics(nn.Module):
    """ Defines the Constrained Hamiltonian dynamics given a Hamiltonian and
    gradients of constraints.

    Args:
        V: A callable function that takes in x and returns the potential
        DPhi:
        wgrad: If True, the dynamics can be backproped.
    """

    def __init__(
        self,
        V,Minv, # And ultimately A
        DPhi: Callable[[Tensor], Tensor],
        shape,
        wgrad: bool = True,
    ):
        super().__init__()
        self.V = V
        self.Minv = Minv
        self.DPhi = DPhi
        self.shape = shape
        self.wgrad = wgrad
        self.nfe = 0


    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        """ Computes a batch of `NxD` time derivatives of the state `z` at time `t`
        Args:
            t: Scalar Tensor of the current time
            z: bs x 2nd Tensor of the bs different states in D=2nd dimensions
        """
        assert (t.ndim == 0) and (z.ndim == 2)
        bs,n,d = z.shape[0],*self.shape
        self.nfe += 1
        with torch.enable_grad():
            x, v = z.reshape(bs,2,n,d).unbind(dim=1) # (bs,n,d)
            x = torch.zeros_like(x, requires_grad=True) + x
            dV = torch.autograd.grad(self.V(x).sum(),x,create_graph=self.wgrad)[0]
            DPhi = self.DPhi(x,v) # (bs,2,n,d,2,C)
            G = DPhi[:,0,:,:,0,:].reshape(bs,n*d,-1) # (bs,nd,C)
            Gdot = DPhi[:,0,:,:,1,:].reshape(bs,n*d,-1) # (bs,nd,C)
            MinvG = self.Minv(G.reshape(bs,n,-1)).reshape(G.shape) # (bs,nd,C)
            GTMinvG = G.permute(0,2,1)@MinvG # (bs, C, C)
            f =  self.Minv(dV).reshape(bs,n*d) # (bs,nd)
            GTf = (G.permute(0,2,1)@f.unsqueeze(-1)).squeeze(-1) # (bs,C)
            violation = (Gdot.permute(0,2,1)@v.reshape(bs,n*d,1)).squeeze(-1) # (bs,C)
            lambdas = (MinvG@torch.solve((GTf+violation).unsqueeze(-1),GTMinvG)[0]).squeeze(-1) # (bs,nd)
            vdot = f - lambdas # (bs,nd)
            dynamics = torch.cat([v.reshape(bs,n*d),vdot],dim=-1) # (bs,2nd)
        return dynamics
