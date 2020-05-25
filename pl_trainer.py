from oil.datasetup.datasets import split_dataset
from oil.utils.utils import FixedNumpySeed

import pytorch_lightning as pl

import sys
import csv
import io
import os
import argparse

import torch
from torch.utils.data import DataLoader
from torch import Tensor

import wandb
import PIL

import numpy as np
from biases.systems.chain_pendulum import ChainPendulum
from biases.systems.rotor import Rotor
from biases.systems.magnet_pendulum import MagnetPendulum
from biases.systems.gyroscope import Gyroscope
from biases.models.constrained_hnn import CHNN, CHLC
from biases.models.constrained_lnn import CLNN, CLLC
from biases.models.hnn import HNN
from biases.models.lnn import LNN, DeLaN
from biases.models.nn import NN, DeltaNN
from biases.datasets import RigidBodyDataset
from biases.systems.rigid_body import rigid_Phi, project_onto_constraints


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


def collect_tensors(field, outputs):
    res = torch.stack([log[field] for log in outputs], dim=0)
    if res.ndim == 1:
        return res
    else:
        return res.flatten(0, 1)


def fig_to_img(fig):
    with io.BytesIO() as buf:
        fig.savefig(buf, format="png")
        buf.seek(0)
        img = wandb.Image(PIL.Image.open(buf))
    return img


class DynamicsModel(pl.LightningModule):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()

        euclidean = hparams.network_class not in [
            "NN",
            "LNN",
            "HNN",
            "DeLaN",
        ]  # TODO: try NN in euclideana
        vars(hparams).update(euclidean=euclidean)

        body = str_to_class(hparams.body_class)(*hparams.body_args)
        vars(hparams).update(dt=body.dt, integration_time=body.integration_time)

        train_dataset = str_to_class(hparams.dataset_class)(
            n_systems=hparams.n_train_systems,
            regen=hparams.regen,
            chunk_len=hparams.chunk_len,
            body=body,
            angular_coords=not euclidean,
            seed=hparams.seed,
            mode="train",
            n_subsample=hparams.n_train,
        )
        val_dataset = str_to_class(hparams.dataset_class)(
            n_systems=hparams.n_val,
            regen=hparams.regen,
            chunk_len=hparams.chunk_len,
            body=body,
            angular_coords=not euclidean,
            seed=hparams.seed + 1,
            mode="val",
        )
        test_dataset = str_to_class(hparams.dataset_class)(
            n_systems=hparams.n_test,
            regen=hparams.regen,
            chunk_len=hparams.chunk_len,
            body=body,
            angular_coords=not euclidean,
            seed=hparams.seed + 2,
            mode="test",
        )

        datasets = {"train": train_dataset, "val": val_dataset, "test": test_dataset}
        splits = {
            "train": hparams.n_train,
            "val": hparams.n_val,
            "test": hparams.n_test,
        }

        net_cfg = {
            "dof_ndim": body.d if euclidean else body.D,
            "angular_dims": body.angular_dims,
            "hidden_size": hparams.n_hidden,
            "num_layers": hparams.n_layers,
            "wgrad": True,
        }
        vars(hparams).update(**net_cfg)

        model = str_to_class(hparams.network_class)(G=body.body_graph, **net_cfg)

        self.hparams = hparams
        self.model = model
        self.body = body
        self.datasets = datasets
        self.splits = splits
        self.batch_sizes = dict(splits)
        self.batch_sizes["train"] = min(
            self.hparams.batch_size, self.batch_sizes["train"]
        )
        self.test_log = None

    def forward(self):
        raise RuntimeError("This module should not be called")

    def rollout(self, z0, ts, tol, method="rk4"):
        # z0: (N x 2 x n_dof x dimensionality of each degree of freedom) sized
        # ts: N x T Tensor representing the time points of true_zs
        # true_zs:  N x T x 2 x n_dof x d sized Tensor
        pred_zs = self.model.integrate(z0, ts, tol=tol, method=method)
        return pred_zs

    def trajectory_mae(self, pred_zts, true_zts):
        return (pred_zts - true_zts).abs().mean()

    def training_step(self, batch: Tensor, batch_idx: int):
        (z0, ts), zts = batch
        # Assume all ts are equally spaced and dynamics is time translation invariant
        ts = ts[0] - ts[0, 0]  # Start ts from 0
        pred_zs = self.rollout(z0, ts, tol=self.hparams.tol, method="rk4")
        loss = self.trajectory_mae(pred_zs, zts)

        logs = {
            "train/trajectory_mae": loss.detach(),
            "train/nfe": self.model.nfe,
        }
        return {
            "loss": loss,
            "log": logs,
        }

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx, integration_factor=0.5)

    def validation_epoch_end(self, outputs):
        log, save = self._collect_test_steps(outputs)
        log = {f"validation/{k}": v for k, v in log.items()}
        return {"val_loss": log["validation/trajectory_mae"], "log": log}

    def test_step(self, batch, batch_idx, integration_factor=1.0):
        (z0, ts), zts = batch
        # Assume all ts are equally spaced and dynamics is time translation invariant
        ts = ts[0] - ts[0, 0]  # Start ts from 0
        pred_zs = self.rollout(z0, ts, tol=self.hparams.tol, method="dopri5")
        loss = self.trajectory_mae(pred_zs, zts)

        (
            pred_zts,
            true_zts,
            pert_zts,
            rel_err_pred_true,
            abs_err_pred_true,
            rel_err_pert_true,
            abs_err_pert_true,
        ) = self.compare_rollouts(
            z0,
            integration_factor * self.body.integration_time,
            self.body.dt,
            self.hparams.tol,
        )
        pred_zts_true_energy = self.true_energy(pred_zts)
        true_zts_true_energy = self.true_energy(true_zts)
        pert_zts_true_energy = self.true_energy(pert_zts)

        return {
            "trajectory_mae": loss.detach(),
            "pred_zts": pred_zts.detach(),
            "true_zts": true_zts.detach(),
            "pert_zts": pert_zts.detach(),
            "rel_err_pred_true": rel_err_pred_true.detach(),
            "abs_err_pred_true": abs_err_pred_true.detach(),
            "rel_err_pert_true": rel_err_pert_true.detach(),
            "abs_err_pert_true": abs_err_pert_true.detach(),
            "pred_zts_true_energy": pred_zts_true_energy.detach(),
            "true_zts_true_energy": true_zts_true_energy.detach(),
            "pert_zts_true_energy": pert_zts_true_energy.detach(),
        }

    def _collect_test_steps(self, outputs):
        loss = collect_tensors("trajectory_mae", outputs).mean(0).item()
        # Average errors across batches
        rel_err_pred_true = (collect_tensors("rel_err_pred_true", outputs)) + 1e-8
        abs_err_pred_true = (collect_tensors("abs_err_pred_true", outputs)) + 1e-8
        rel_err_pert_true = (collect_tensors("rel_err_pert_true", outputs)) + 1e-8
        abs_err_pert_true = (collect_tensors("abs_err_pert_true", outputs)) + 1e-8
        # average of integration of log errs
        int_rel_err_pred_true = self.integrate_curve(
            rel_err_pred_true.log(), dt=self.body.dt
        ).mean(0)
        int_abs_err_pred_true = self.integrate_curve(
            abs_err_pred_true.log(), dt=self.body.dt
        ).mean(0)
        int_rel_err_pert_true = self.integrate_curve(
            rel_err_pert_true.log(), dt=self.body.dt
        ).mean(0)
        int_abs_err_pert_true = self.integrate_curve(
            abs_err_pert_true.log(), dt=self.body.dt
        ).mean(0)

        pred_zts_true_energy = collect_tensors("pred_zts_true_energy", outputs)
        true_zts_true_energy = collect_tensors("true_zts_true_energy", outputs)
        pert_zts_true_energy = collect_tensors("pert_zts_true_energy", outputs)

        int_pred_true_energy = self.integrate_curve(
            pred_zts_true_energy, dt=self.body.dt
        ).mean(0)
        int_true_true_energy = self.integrate_curve(
            true_zts_true_energy, dt=self.body.dt
        ).mean(0)
        int_pert_true_energy = self.integrate_curve(
            pert_zts_true_energy, dt=self.body.dt
        ).mean(0)
        log = {
            "trajectory_mae": loss,
            "int_rel_err_pred_true": int_rel_err_pred_true,
            "int_abs_err_pred_true": int_abs_err_pred_true,
            "int_rel_err_pert_true": int_rel_err_pert_true,
            "int_abs_err_pert_true": int_abs_err_pert_true,
            "int_pred_true_energy": int_pred_true_energy,
            "int_true_true_energy": int_true_true_energy,
            "int_pert_true_energy": int_pert_true_energy,
        }
        pred_zts = collect_tensors("pred_zts", outputs)
        true_zts = collect_tensors("true_zts", outputs)
        pert_zts = collect_tensors("pert_zts", outputs)
        save = {"pred_zts": pred_zts, "true_zts": true_zts, "pert_zts": pert_zts}
        save.update(log)
        return log, save

    def test_epoch_end(self, outputs):
        log, save = self._collect_test_steps(outputs)
        log = {f"test/{k}": v for k, v in log.items()}
        return {"log": log, "test_log": save}

    def compare_rollouts(
        self, z0: Tensor, integration_time: float, dt: float, tol: float, pert_eps=1e-4
    ):
        prev_device = list(self.parameters())[0].device
        prev_dtype = list(self.parameters())[0].dtype
        ts = torch.arange(0.0, integration_time, dt, device=z0.device, dtype=z0.dtype)
        pred_zts = self.rollout(z0, ts, tol, "dopri5")
        bs, Nlong, *rest = pred_zts.shape
        body = self.datasets["test"].body
        if not self.hparams.euclidean:  # convert to euclidean for body to integrate
            z0 = body.body2globalCoords(z0)
            flat_pred = body.body2globalCoords(pred_zts.reshape(bs * Nlong, *rest))

            pred_zts = flat_pred.reshape(bs, Nlong, *flat_pred.shape[1:])

        # (bs, n_steps, 2, n_dof, d)
        true_zts = body.integrate(z0, ts, tol=tol)
        perturbation = pert_eps * torch.randn_like(
            z0
        )  # perturbation does not respect constraints
        z0_perturbed = project_onto_constraints(
            body.body_graph, z0 + perturbation
        )  # project
        pert_zts = body.integrate(z0_perturbed, ts, tol=tol)

        sq_diff_pred_true = (pred_zts - true_zts).pow(2).sum((2, 3, 4))
        sq_diff_pert_true = (true_zts - pert_zts).pow(2).sum((2, 3, 4))
        sq_sum_pred_true = (pred_zts + true_zts).pow(2).sum((2, 3, 4))
        sq_sum_pert_true = (true_zts + pert_zts).pow(2).sum((2, 3, 4))

        # (bs, n_step)
        rel_err_pred_true = sq_diff_pred_true.div(sq_sum_pred_true).sqrt()
        abs_err_pred_true = sq_diff_pred_true.sqrt()
        rel_err_pert_true = sq_diff_pert_true.div(sq_sum_pert_true).sqrt()
        abs_err_pert_true = sq_diff_pert_true.sqrt()

        self.to(prev_dtype)
        self.to(prev_device)
        return (
            pred_zts,
            true_zts,
            pert_zts,
            rel_err_pred_true,
            abs_err_pred_true,
            rel_err_pert_true,
            abs_err_pert_true,
        )

    def true_energy(self, zs):
        N, T = zs.shape[:2]
        q, qdot = zs.chunk(2, dim=2)
        p = self.body.M @ qdot
        zs = torch.cat([q, p], dim=2)
        energy = self.body.hamiltonian(None, zs.reshape(N * T, -1))
        return energy.reshape(N, T)

    def integrate_curve(self, y, t=None, dt=1.0, axis=-1):
        # If y is error, then we want to minimize the returned result
        if torch.is_tensor(y):
            y = y.detach().cpu().numpy()
        return np.trapz(y, t, dx=dt, axis=axis)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams.optimizer_class)(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        if self.hparams.no_lr_sched:
            return optimizer
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.n_epochs, eta_min=0.0,
            )
            return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_sizes["train"],
            pin_memory=torch.cuda.is_available(),
            num_workers=0,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["val"],
            batch_size=self.batch_sizes["val"],
            pin_memory=torch.cuda.is_available(),
            num_workers=0,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_sizes["test"],
            pin_memory=torch.cuda.is_available(),
            num_workers=0,
            shuffle=False,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch-size", type=int, default=200, help="Batch size")
        parser.add_argument(
            "--body-class",
            type=str,
            help="Class name of physical system",
            required=True,
        )
        parser.add_argument(
            "--body-args",
            help="Arguments to initialize physical system separated by spaces",
            nargs="*",
            type=int,
            default=[],
        )
        parser.add_argument(
            "--no-lr-sched",
            action="store_true",
            default=False,
            help="Turn off cosine annealing for learing rate",
        )
        parser.add_argument(
            "--chunk-len",
            type=int,
            default=5,
            help="Length of each chunk of training trajectory",
        )
        parser.add_argument(
            "--dataset-class",
            type=str,
            default="RigidBodyDataset",
            help="Dataset class",
        )
        ########## dt and integration_time are now attributes of body ###################
        # parser.add_argument(
        #     "--dt", type=float, default=1e-1, help="Timestep size in generated data"
        # )
        # parser.add_argument(
        #     "--integration-time",
        #     type=float,
        #     default=10.0,
        #     help="Amount of time to integrate for in generating training trajectories",
        # )
        #################################################################################
        parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
        parser.add_argument(
            "--n-test", type=int, default=100, help="Number of test trajectories"
        )
        parser.add_argument(
            "--n-train", type=int, default=800, help="Number of train trajectories"
        )
        parser.add_argument(
            "--n-val", type=int, default=100, help="Number of validation trajectories"
        )
        parser.add_argument(
            "--network-class",
            type=str,
            help="Dynamics network",
            choices=[
                "NN",
                "DeltaNN",
                "HNN",
                "LNN",
                "DeLaN",
                "CHNN",
                "CLNN",
                "CHLC",
                "CLLC",
            ],
        )
        parser.add_argument(
            "--n-epochs", type=int, default=2000, help="Number of training epochs"
        )
        parser.add_argument(
            "--n-hidden", type=int, default=200, help="Number of hidden units"
        )
        parser.add_argument(
            "--n-layers", type=int, default=3, help="Number of hidden layers"
        )
        parser.add_argument(
            "--n-train-systems", type=int, default=10000, help="Number of hidden layers"
        )
        parser.add_argument(
            "--optimizer_class", type=str, default="AdamW", help="Optimizer",
        )
        parser.add_argument(
            "--seed", type=int, default=0, help="Seed used to generate dataset",
        )
        parser.add_argument(
            "--tol",
            type=float,
            default=1e-7,
            help="Tolerance for numerical intergration",
        )
        parser.add_argument(
            "--regen",
            action="store_true",
            default=False,
            help="Forcibly regenerate training data",
        )
        parser.add_argument(
            "--weight-decay", type=float, default=1e-4, help="Weight decay",
        )
        return parser


class SaveTestLogCallback(pl.Callback):
    def on_test_end(self, trainer, pl_module):
        if type(trainer.logger) == WandbLogger:
            save_dir = os.path.join(trainer.logger.experiment.dir, "test_log.pt")
            if "test_log" in trainer.callback_metrics:
                # Use torch.save in case we want to save pytorch tensors or modules
                torch.save(trainer.callback_metrics["test_log"], save_dir)


def parse_misc():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gradient-clip-val", type=float, default=0, help="Threshold for 2-norm of gradient. 0 is off")
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Debug code by running 1 batch of train, val, and test.",
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="",
        help="Directory to save files from this experiment",
    )
    parser.add_argument(
        "--n-epochs-per-val",
        type=int,
        default=100,
        help="Number of training epochs per validation step",
    )
    parser.add_argument("--n-gpus", type=int, default=1, help="Number of training GPUs")
    parser.add_argument(
        "--tags", type=str, nargs="*", default=None, help="Experiment tags"
    )
    parser.add_argument("--track-grad-norm", type=int, default=-1, help="Log gradient norms")
    return parser


if __name__ == "__main__":
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.callbacks import LearningRateLogger

    parser = parse_misc()
    parser = DynamicsModel.add_model_specific_args(parser)
    hparams = parser.parse_args()

    dynamics_model = DynamicsModel(hparams=hparams)

    # create experiment directory
    if hparams.exp_dir == "":
        exp_dir = os.path.join(
            os.getcwd(),
            "experiments",
            f"{dynamics_model.body.__repr__()}",
            f"{hparams.network_class}",
        )
    else:
        exp_dir = hparams.exp_dir
    # Note that this args is shared with the model's hparams so it will be saved
    vars(hparams).update(exp_dir=exp_dir)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        print("Directory ", exp_dir, " Created ")
    else:
        print("Directory ", exp_dir, " already exists")

    logger = WandbLogger(
        save_dir=exp_dir, project="constrained-pnns", log_model=True, tags=hparams.tags
    )
    ckpt_dir = os.path.join(
        logger.experiment.dir, logger.name, f"version_{logger.version}", "checkpoints",
    )
    if hparams.no_lr_sched:
        callbacks = [SaveTestLogCallback()]
    else:
        callbacks = [LearningRateLogger(), SaveTestLogCallback()]
    vars(hparams).update(
        check_val_every_n_epoch=hparams.n_epochs_per_val,
        fast_dev_run=hparams.debug,
        gpus=hparams.n_gpus,
        max_epochs=hparams.n_epochs,
        ckpt_dir=ckpt_dir,
    )

    # record human-readable hparams as csv
    with open(os.path.join(logger.experiment.dir, "args.csv"), "w") as csvfile:
        args_dict = vars(hparams)  # convert to dict with new copy
        writer = csv.DictWriter(csvfile, fieldnames=args_dict.keys())
        writer.writeheader()
        writer.writerow(args_dict)

    trainer = Trainer.from_argparse_args(hparams, callbacks=callbacks, logger=logger)

    trainer.fit(dynamics_model)

    with torch.no_grad():
        trainer.test()

    # ckpt_path = os.path.join(ckpt_dir, f"epoch={args.n_epochs - 1}.ckpt")
    # probably remove logger when resuming since it's a finished experiment
    # loaded_trainer = Trainer(
    #    resume_from_checkpoint=ckpt_path, callbacks=callbacks, logger=logger
    # )
    # loaded_model = DynamicsModel.load_from_checkpoint(ckpt_path)
