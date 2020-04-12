import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable
from lie_conv.utils import export


@export
class HamiltonianDynamics(nn.Module):
    """ Defines the dynamics given a Hamiltonian.

    Args:
        H: A callable function that takes in q and p concatenated together and returns H(q, p)
        wgrad: If True, the dynamics can be backproped.
    """

    def __init__(self, H: Callable[[Tensor, Tensor], Tensor], wgrad: bool = False):
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
        return J(dH.unsqueeze(-1)).squeeze(-1)


@export
class ConstrainedHamiltonianDynamics(nn.Module):
    """ Defines the Constrained Hamiltonian dynamics given a Hamiltonian and
    gradients of constraints.

    Args:
        H: A callable function that takes in q and p and returns H(q, p)
        DPhi: Matrix containing gradients of constraints
        wgrad: If True, the dynamics can be backproped.
    """

    def __init__(self, H, DPhi, wgrad=False):
        super().__init__()
        self.H = H
        self.DPhi = DPhi
        self.wgrad = wgrad
        self.nfe = 0

    def forward(self, t, z):
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
        return P(J(dH.unsqueeze(-1))).squeeze(-1)


def J(M):
    """ applies the J matrix to another matrix M.
        input: M (*,2nd,b), output: J@M (*,2nd,b)"""
    *star, D, b = M.shape
    JM = torch.cat([M[..., D // 2 :, :], -M[..., : D // 2, :]], dim=-2)
    return JM


def Proj(DPhi):
    def _P(M):
        DPhiT = DPhi.transpose(-1, -2)
        reg = 0  # 1e-4*torch.eye(DPhi.shape[-1],dtype=DPhi.dtype,device=DPhi.device)[None]
        X, _ = torch.solve(DPhiT @ M, DPhiT @ J(DPhi) + reg)
        return M - J(DPhi @ X)

    return _P


def EuclideanT(p, Minv, function=False):
    """ Shape (bs,n,d), and (bs,n,n),
        standard \sum_n pT Minv p/2 kinetic energy"""
    if function:
        return (p * Minv(p)).sum(-1).sum(-1) / 2
    else:
        return (p * (Minv @ p)).sum(-1).sum(-1) / 2
