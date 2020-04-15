import torch
import torch.nn as nn
from torchdiffeq import odeint
from torch.autograd import grad
from torch import Tensor
from typing import Callable
from oil.utils.utils import export


@export
class LagrangianDynamics(nn.Module):
    """ Defines the dynamics given a Lagrangian.

    Args:
        L: A callable function that takes in q and qdot concatenated together and returns L(q, qdot)
        wgrad: If True, the dynamics can be backproped.
    """

    def __init__(self, L: Callable[[Tensor, Tensor], Tensor], wgrad: bool = True):
        super().__init__()
        self.L = L
        self.wgrad = wgrad

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        """ Computes a batch of `NxD` time derivatives of the state `z` at time `t`
        Args:
            t: Scalar Tensor of the current time
            z: N x D Tensor of the N different states in D dimensions
        """
        assert (t.ndim == 0) and (z.ndim == 2)
        D = z.shape[-1]
        d = D // 2
        with torch.enable_grad():
            q = z[..., :d]
            v = z[..., d:]
            # we build our graph through (q, v) -> z -> L so that we can take gradients wrt q, v, and z
            # if instead we build it through z -> (q, v) and z -> L, then we can't take gradients wrt q and v
            q = q + torch.zeros_like(q, requires_grad=True)
            v = v + torch.zeros_like(v, requires_grad=True)
            z = torch.cat([q, v], dim=-1)
            L = self.L(t, z).sum()  # elements in mb are independent, gives mb gradients
            dL_dz = grad(L, z, create_graph=True)[0]  # gradient
            dL_dq = dL_dz[..., :d]
            dL_dv = dL_dz[..., d:]
            Fv = -grad(
                (dL_dq * v.detach()).sum(), v, create_graph=True, allow_unused=True
            )[0]
            # elements in mb are independent, gives mb gradients
            eye = torch.eye(d, device=z.device, dtype=z.dtype)
            M = torch.stack(
                [
                    grad((dL_dv * eye[i]).sum(), v, create_graph=self.wgrad)[0]
                    for i in range(d)
                ],
                dim=-1,
            )
            F = (dL_dq + Fv).unsqueeze(-1)
            a = torch.solve(F, M)[0].squeeze(-1)
            dynamics = torch.cat([v, a], dim=-1)  # +Fv#
        return dynamics


def PendulumLagrangian(z: Tensor):
    q = z[..., 0]
    v = z[..., 1]
    return v * v / 2 + (q.cos() - 1)

@export
def LagrangianFlow(
    L: Callable[[Tensor, Tensor], Tensor], z0: Tensor, T: Tensor, higher: bool = False
) -> Tensor:
    def dynamics(t: Tensor, z: Tensor):
        return LagrangianDynamics(L, higher)(t, z)

    return odeint(dynamics, z0, T, rtol=1e-6).permute(1, 0, 2)
