import torch
import torch.nn as nn


class HamiltonianDynamics(nn.Module):
    """ Defines the dynamics given a hamiltonian. If wgrad=True, the dynamics can be backproped."""

    def __init__(self, H, wgrad=False):
        super().__init__()
        self.H = H
        self.wgrad = wgrad
        self.nfe = 0

    def forward(self, t, z):
        self.nfe += 1
        with torch.enable_grad():
            z = torch.zeros_like(z, requires_grad=True) + z
            H = self.H(t, z).sum()  # elements in mb are independent, gives mb gradients
            dH = torch.autograd.grad(H, z, create_graph=self.wgrad)[0]  # gradient
        return J(dH.unsqueeze(-1)).squeeze(-1)


class ConstrainedHamiltonianDynamics(nn.Module):
    """ Defines the dynamics given a hamiltonian. If wgrad=True, the dynamics can be backproped."""

    def __init__(self, H, DPhi, wgrad=False):
        super().__init__()
        self.H = H
        self.DPhi = DPhi
        self.wgrad = wgrad
        self.nfe = 0

    def forward(self, t, z):
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
