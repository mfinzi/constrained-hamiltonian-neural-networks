import torch
import torch.nn as nn
from oil.utils.utils import Eval
from oil.model_trainers import Trainer
from oil.utils.utils import export
import numpy as np


@export
class IntegratedDynamicsTrainer(Trainer):
    """ Model should specify the dynamics, mapping from t,z -> dz/dt"""

    def __init__(self, *args, tol=1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self.hypers["tol"] = tol
        self.num_mbs = 0

    def loss(self, minibatch):
        """ Standard cross-entropy loss """
        (z0, ts), true_zs = minibatch
        pred_zs = self.model.integrate(z0, ts[0], tol=self.hypers["tol"])
        self.num_mbs += 1
        return (pred_zs - true_zs).pow(2).mean()

    def metrics(self, loader):
        mse = lambda mb: self.loss(mb).cpu().data.numpy()
        return {"MSE": self.evalAverageMetrics(loader, mse)}

    def logStuff(self, step, minibatch=None):
        self.logger.add_scalars(
            "info", {"nfe": self.model.nfe / (max(self.num_mbs, 1e-3))}, step
        )
        super().logStuff(step, minibatch)

    def test_rollouts(self, angular_to_euclidean=False, pert_eps=1e-4):
        self.model.cpu().double()
        dataloader = self.dataloaders["test"]
        rel_errs = []
        pert_rel_errs = []
        with Eval(self.model), torch.no_grad():
            for mb in dataloader:
                z0, T = mb[0]  # assume timesteps evenly spaced for now
                z0 = z0.cpu().double()
                T = T[0]
                dT = (T[-1] - T[0]) / len(T)
                long_T = dT * torch.arange(50 * len(T)).to(z0.device, z0.dtype)
                zt_pred = self.model.integrate(z0, long_T)
                bs, Nlong, *rest = zt_pred.shape
                # add conversion from angular to euclidean
                body = dataloader.dataset.body
                if angular_to_euclidean:
                    z0 = body.body2globalCoords(z0)
                    flat_pred = body.body2globalCoords(
                        zt_pred.reshape(bs * Nlong, *rest)
                    )
                    zt_pred = flat_pred.reshape(bs, Nlong, *flat_pred.shape[1:])
                zt = dataloader.dataset.body.integrate(z0, long_T)
                perturbation = pert_eps * torch.randn_like(z0)
                zt_pert = dataloader.dataset.body.integrate(z0 + perturbation, long_T)
                # (bs,T,2,n,2)
                rel_error = ((zt_pred - zt) ** 2).sum(-1).sum(-1).sum(-1).sqrt() / (
                    (zt_pred + zt) ** 2
                ).sum(-1).sum(-1).sum(-1).sqrt()
                rel_errs.append(rel_error)
                pert_rel_error = ((zt_pert - zt) ** 2).sum(-1).sum(-1).sum(
                    -1
                ).sqrt() / ((zt_pert + zt) ** 2).sum(-1).sum(-1).sum(-1).sqrt()
                pert_rel_errs.append(pert_rel_error)
            rel_errs = torch.cat(rel_errs, dim=0)  # (D,T)
            pert_rel_errs = torch.cat(pert_rel_errs, dim=0)  # (D,T)
            both = torch.stack([rel_errs, pert_rel_errs], dim=-1)  # (D,T,2)
        return both


def logspace(a, b, k):
    return np.exp(np.linspace(np.log(a), np.log(b), k))


from torch.nn.utils import spectral_norm


def add_spectral_norm(module):
    if isinstance(
        module,
        (
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
        ),
    ):
        spectral_norm(module, dim=1)
        # print("SN on conv layer: ",module)
    elif isinstance(module, nn.Linear):
        spectral_norm(module, dim=0)
        # print("SN on linear layer: ",module)
