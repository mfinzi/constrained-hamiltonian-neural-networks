import torch
import torch.nn as nn
from torchdiffeq import odeint  # odeint_adjoint as odeint
import numpy as np
from lie_conv.utils import export


@export
class LyapunovDynamics(nn.Module):
    def __init__(self, F):
        super().__init__()
        self.F = F

    def forward(self, t, xqr):
        """ Computes a batch of `NxD` time derivatives of the state `xqr` at time `t`
        Args:
            t: Scalar Tensor of the current time
            z: N x D Tensor of the N different states in D dimensions
        """
        assert (t.ndim == 0) and (xqr.ndim == 2)
        n = (xqr.shape[-1] - 1) // 2
        with torch.enable_grad():
            x = xqr[..., :n] + torch.zeros_like(xqr[..., :n], requires_grad=True)
            q = xqr[..., n : 2 * n]
            xdot = self.F(t, x)
            DFq = vjp(xdot, x, q)
        qDFq = (q * DFq).sum(-1, keepdims=True)
        qdot = DFq - q * qDFq
        lrdot = qDFq
        xqrdot = torch.cat([xdot, qdot, lrdot], dim=-1)
        return xqrdot


def MLE(xt, ts, F, v0=None):
    """ Computes the Maximal Lyapunov exponent using the Power iteration.
        inputs: trajectory [xt (T,*)] dynamics [F] """
    v = torch.randn_like(xt[0]) if v0 is None else v0
    dt = ts[1] - ts[0]
    exps = []
    for i, x in enumerate(xt):
        # for j in range(5):
        x = torch.zeros_like(x, requires_grad=True) + x.detach()
        y = F(ts[i], x[None])[0]
        u = v + vjp(y, x, v).detach() * dt
        # u  = v+ dt*(F(ts[i],(x+1e-7*v)[None])[0]-F(ts[i],x[None])[0])/(1e-7)
        r = (u ** 2).sum().sqrt().detach()
        v = u / r  # P((u/r)[None,:,None])[0,:,0]
        exps += [r.log().item() / dt]  # (1/i)*(r.log() - exp)
        # print(r.log()/(100/5000))
    return np.array(exps)  # ,u


def MLE2(x0, F, ts, **kwargs):
    with torch.no_grad():
        LD = LyapunovDynamics(F)
        x0 = x0.reshape(x0.shape[0], -1)
        q0 = torch.randn_like(x0)
        q0 /= (q0 ** 2).sum(-1, keepdims=True).sqrt()
        lr0 = torch.zeros(x0.shape[0], 1, dtype=x0.dtype, device=x0.device)
        Lx0 = torch.cat([x0, q0, lr0], dim=-1)
        Lxt = odeint(LD, Lx0, ts, **kwargs)
        maximal_exponent = Lxt  # [...,-1]
    return maximal_exponent


def jvp(y, x, v):
    with torch.enable_grad():
        Jv = torch.autograd.grad(y, [x], [v])[0]
    return Jv


def vjp(y, x, v):
    # Following the trick from https://j-towns.github.io/2017/06/12/A-new-trick.html
    with torch.enable_grad():
        u = torch.ones_like(
            y, requires_grad=True
        )  # Dummy variable (could take any value)
        Ju = torch.autograd.grad(y, [x], [u], create_graph=True)[0]
        vJ = torch.autograd.grad(Ju, [u], [v])[0]
    return vJ
