import torch
import torch.nn as nn
from oil.utils.utils import Eval
from oil.model_trainers import Trainer
from oil.utils.utils import export
import numpy as np
from biases.systems.rigid_body import project_onto_constraints

@export
class IntegratedDynamicsTrainer(Trainer):
    """ Model should specify the dynamics, mapping from t,z -> dz/dt"""

    def __init__(self, *args, tol=1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self.hypers["tol"] = tol
        self.num_mbs = 0
        self.text_to_add=np.arange(1)
    def loss(self, minibatch):
        """ Standard cross-entropy loss """
        (z0, ts), true_zs = minibatch
        pred_zs = self.model.integrate(z0, ts[0], tol=self.hypers["tol"])
        self.num_mbs += 1
        self.text_to_add = true_zs[0,:,0,:].cpu().detach().data.numpy()#true_zs[:,:,1,2].abs().max().cpu().detach().data.numpy()
        return (pred_zs - true_zs).abs().mean()

    def metrics(self, loader):
        mae = lambda mb: self.loss(mb).cpu().data.numpy()
        return {"MAE": self.evalAverageMetrics(loader, mae)}

    def logStuff(self, step, minibatch=None):
        self.logger.add_scalars(
            "info", {"nfe": self.model.nfe / (max(self.num_mbs, 1e-3))}, step
        )
        #print(self.text_to_add)
        super().logStuff(step, minibatch)

    def test_rollouts(self, angular_to_euclidean=False, pert_eps=1e-4):
        #self.model.cpu().double()
        dataloader = self.dataloaders["test"]
        rel_errs = []
        pert_rel_errs = []
        with Eval(self.model), torch.no_grad():
            for mb in dataloader:
                z0, T = mb[0]  # assume timesteps evenly spaced for now
                #z0 = z0.cpu().double()
                T = T[0]
                body = dataloader.dataset.body
                long_T = body.dt * torch.arange(body.integration_time//body.dt).to(z0.device, z0.dtype)
                zt_pred = self.model.integrate(z0, long_T,tol=1e-6,method='dopri5')
                bs, Nlong, *rest = zt_pred.shape
                # add conversion from angular to euclidean
                
                if angular_to_euclidean:
                    z0 = body.body2globalCoords(z0)
                    flat_pred = body.body2globalCoords(zt_pred.reshape(bs * Nlong, *rest))
                    zt_pred = flat_pred.reshape(bs, Nlong, *flat_pred.shape[1:])
                zt = dataloader.dataset.body.integrate(z0, long_T)
                perturbation = pert_eps * torch.randn_like(z0) # perturbation does not respect constraints
                z0_perturbed = project_onto_constraints(body.body_graph,z0 + perturbation,tol=1e-5) #project
                zt_pert = body.integrate(z0_perturbed, long_T)
                # (bs,T,2,n,2)
                rel_error = ((zt_pred - zt) ** 2).sum(-1).sum(-1).sum(-1).sqrt() / (
                    (zt_pred + zt) ** 2
                ).sum(-1).sum(-1).sum(-1).sqrt()
                rel_errs.append(rel_error)
                pert_rel_error = ((zt_pert - zt) ** 2).sum(-1).sum(-1).sum(-1 \
                ).sqrt() / ((zt_pert + zt) ** 2).sum(-1).sum(-1).sum(-1).sqrt()
                pert_rel_errs.append(pert_rel_error)
            rel_errs = torch.cat(rel_errs, dim=0)  # (D,T)
            pert_rel_errs = torch.cat(pert_rel_errs, dim=0)  # (D,T)
            both = torch.stack([rel_errs, pert_rel_errs], dim=-1)  # (D,T,2)
        return both


def logspace(a, b, k):
    return np.exp(np.linspace(np.log(a), np.log(b), k))
