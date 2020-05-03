from oil.datasetup.datasets import split_dataset
from oil.utils.utils import FixedNumpySeed

import pytorch_lightning as pl

import matplotlib.pyplot as plt

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


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


def collect_tensors(field, outputs):
    return torch.stack([log[field] for log in outputs], dim=0)


def fig_to_img(fig):
    with io.BytesIO() as buf:
        fig.savefig(buf, format="png")
        buf.seek(0)
        img = wandb.Image(PIL.Image.open(buf))
    return img


class DynamicsModel(pl.LightningModule):
    def __init__(
        self,
        hparams: dict,
        chunk_len: int,
        batch_size: int,
        body_class: str,
        body_args: list,
        dataset_class: str,
        dt: float,
        integration_time: float,
        lr: float,
        no_lr_sched: bool,
        n_epochs: int,
        n_hidden: int,
        n_layers: int,
        n_train: int,
        n_val: int,
        n_test: int,
        network_class: str,
        optimizer_class: str,
        regen: bool,
        seed: int,
        tol: float,
        weight_decay: float,
        **unused_kwargs,
    ):
        super().__init__()

        print("Unused kwargs:", unused_kwargs)

        euclidean = network_class not in [
            "NN",
            "LNN",
            "HNN",
        ]  # TODO: try NN in euclideana
        hparams.update(euclidean=euclidean)

        body = str_to_class(body_class)(*body_args)

        dataset = str_to_class(dataset_class)(
            n_systems=n_train + n_val + n_test,
            regen=regen,
            chunk_len=chunk_len,
            body=body,
            dt=dt,
            integration_time=integration_time,
            angular_coords=not euclidean,
        )
        splits = {"train": n_train, "val": n_val, "test": n_test}
        with FixedNumpySeed(seed):
            datasets = split_dataset(dataset, splits)

        net_cfg = {
            "dof_ndim": body.d if euclidean else body.D,
            "angular_dims": body.angular_dims,
            "hidden_size": n_hidden,
            "num_layers": n_layers,
            "wgrad": True,
        }
        hparams.update(**net_cfg)

        model = str_to_class(network_class)(G=body.body_graph, **net_cfg)

        self.hparams = hparams
        self.model = model
        self.body = body
        self.datasets = datasets
        self.splits = splits
        self.batch_sizes = {k: min(batch_size, v) for k, v in splits.items()}

    def forward(self):
        raise RuntimeError("This module should not be called")

    def rollout(self, z0, ts, tol):
        # z0: (N x 2 x n_dof x dimensionality of each degree of freedom) sized
        # ts: N x T Tensor representing the time points of true_zs
        # true_zs:  N x T x 2 x n_dof x d sized Tensor
        pred_zs = self.model.integrate(z0, ts, tol=tol)
        return pred_zs

    def trajectory_mse(self, pred_zts, true_zts):
        return (pred_zts - true_zts).pow(2).mean()

    def training_step(self, batch: Tensor, batch_idx: int):
        (z0, ts), zts = batch
        # Assume all ts are equally spaced and dynamics is time translation invariant
        ts = ts[0] - ts[0, 0]  # Start ts from 0
        pred_zs = self.rollout(z0, ts, tol=self.hparams["tol"])
        loss = self.trajectory_mse(pred_zs, zts)

        logs = {
            "train/trajectory_mse": loss.detach(),
            "train/nfe": self.model.nfe,
        }
        return {
            "loss": loss,
            "log": logs,
        }

    def validation_step(self, batch, batch_idx):
        (z0, ts), zts = batch
        # Assume all ts are equally spaced and dynamics is time translation invariant
        ts = ts[0] - ts[0, 0]  # Start ts from 0
        pred_zs = self.rollout(z0, ts, tol=self.hparams["tol"])
        loss = self.trajectory_mse(pred_zs, zts)
        return {"trajectory_mse": loss.detach()}

    def validation_epoch_end(self, outputs):
        loss = collect_tensors("trajectory_mse", outputs).mean(0).item()
        log = {"validation/trajectory_mse": loss}
        return {"val_loss": loss, "log": log}

    def test_step(self, batch, batch_idx):
        (z0, ts), zts = batch
        # Assume all ts are equally spaced and dynamics is time translation invariant
        ts = ts[0] - ts[0, 0]  # Start ts from 0
        pred_zs = self.rollout(z0, ts, tol=self.hparams["tol"])
        loss = self.trajectory_mse(pred_zs, zts)

        batch_trajectory_time = (ts[-1] - ts[0]).item()
        dt = (ts[1] - ts[0]).item()
        pred_zts, true_zts, _, rel_error, abs_error = self.compare_rollouts(
            z0, 50 * batch_trajectory_time, dt, self.hparams["tol"]
        )
        return {
            "trajectory_mse": loss.detach(),
            "rel_error": rel_error.detach(),
            "abs_error": abs_error.detach(),
        }

    def test_epoch_end(self, outputs):
        loss = collect_tensors("trajectory_mse", outputs).mean(0).item()
        # Average errors across batches
        rel_error = collect_tensors("rel_error", outputs).mean((0, 1)).numpy()
        abs_error = collect_tensors("abs_error", outputs).mean((0, 1)).numpy()
        fig, ax = plt.subplots()
        ax.plot(rel_error, label="Relative Error")
        ax.plot(abs_error, label="Absolute Error")
        ax.set(yscale="log", xlabel="Forward prediction (x 0.1 seconds)")
        ax.legend()

        # TODO: save arrays as temporary files and then wandb.save
        # self.logger.experiment.save(rel_error)
        # self.logger.experiment.save(abs_error)

        log = {
            "test/trajectory_mse": loss,
            "test/rollout_error": fig_to_img(fig),
        }
        return {"log": log}

    def compare_rollouts(
        self, z0: Tensor, integration_time: float, dt: float, tol: float, pert_eps=1e-4
    ):
        # Ground truth is in double so we convert model to double
        prev_device = list(self.parameters())[0].device
        prev_dtype = list(self.parameters())[0].dtype
        self.double()
        self.cpu()
        z0 = z0.double().cpu()
        ts = torch.arange(0.0, integration_time, dt, device=z0.device, dtype=z0.dtype)
        pred_zts = self.rollout(z0, ts, tol)

        bs, Nlong, *rest = pred_zts.shape
        body = self.datasets["test"].body
        if not self.hparams["euclidean"]:  # convert to euclidean for body to integrate
            z0 = body.body2globalCoords(z0).to(z0.device)
            flat_pred = body.body2globalCoords(pred_zts.reshape(bs * Nlong, *rest)).to(z0.device)
            pred_zts = flat_pred.reshape(bs, Nlong, *flat_pred.shape[1:])

        # (bs, n_steps, 2, n_dof, d)
        true_zts = body.integrate(z0, ts, tol=tol)

        perturbation = pert_eps * torch.randn_like(z0)
        true_zts_pert = body.integrate(z0 + perturbation, ts, tol=tol)

        sq_diff_from_true = (pred_zts - true_zts).pow(2).sum((2, 3, 4))
        sq_sum_from_true = (pred_zts + true_zts).pow(2).sum((2, 3, 4))

        # (bs, n_step)
        rel_error = sq_diff_from_true.div(sq_sum_from_true).sqrt()
        abs_error = sq_diff_from_true.sqrt()

        # TODO: compute error from pert
        self.to(prev_device)
        self.to(prev_dtype)
        return pred_zts, true_zts, true_zts_pert, rel_error, abs_error

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams["optimizer_class"])(
            self.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        if self.hparams["no_lr_sched"]:
            return optimizer
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams["n_epochs"],
                eta_min=0.,
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


def parse_cmdline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=200, help="Batch size")
    parser.add_argument(
        "--body-class", type=str, help="Class name of physical system", required=True,
    )
    parser.add_argument(
        "--body-args",
        help="Arguments to initialize physical system separated by spaces",
        nargs="*",
        type=int,
        default=[]
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
        "--dataset-class", type=str, default="RigidBodyDataset", help="Dataset class",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Debug code by running 1 batch of train, val, and test.",
    )
    parser.add_argument(
        "--dt", type=float, default=1e-1, help="Timestep size in generated data"
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="",
        help="Directory to save files from this experiment",
    )
    parser.add_argument(
        "--integration-time",
        type=float,
        default=10.0,
        help="Amount of time to integrate for in generating training trajectories",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--n-epochs-per-val",
        type=int,
        default=50,
        help="Number of training epochs per validation step",
    )
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
        choices=["NN", "DeltaNN", "HNN", "LNN", "CHNN", "CLNN", "CHLC", "CLLC"],
    )
    parser.add_argument(
        "--n-epochs", type=int, default=300, help="Number of training epochs"
    )
    parser.add_argument("--n-gpus", type=int, default=1, help="Number of training GPUs")
    parser.add_argument(
        "--n_hidden", type=int, default=200, help="Number of hidden units"
    )
    parser.add_argument(
        "--n-layers", type=int, default=3, help="Number of hidden layers"
    )
    parser.add_argument(
        "--optimizer_class", type=str, default="AdamW", help="Optimizer",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed used to generate dataset",
    )
    parser.add_argument(
        "--tol", type=float, default=1e-4, help="Tolerance for numerical intergration",
    )
    parser.add_argument(
        "--regen",
        action="store_true",
        default=False,
        help="Forcibly regenerate training data",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0., help="Weight decay",
    )
    return parser.parse_args()


if __name__ == "__main__":
    from biases.systems.chain_pendulum import ChainPendulum
    from biases.systems.rotor import Rotor
    from biases.systems.magnet_pendulum import MagnetPendulum
    from biases.systems.gyroscope import Gyroscope
    from biases.models.constrained_hnn import CHNN, CHLC
    from biases.models.constrained_lnn import CLNN, CLLC
    from biases.models.hnn import HNN
    from biases.models.lnn import LNN
    from biases.models.nn import NN, DeltaNN
    from biases.datasets import RigidBodyDataset
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import WandbLogger

    args = parse_cmdline()
    args_dict = dict(vars(args))  # convert to dict with new copy

    dynamics_model = DynamicsModel(hparams=args_dict, **args_dict)

    if args.exp_dir == "":
        exp_dir = "/".join(
            [
                ".",
                "experiments",
                f"{dynamics_model.body.__repr__()}",
                f"{args.network_class}",
            ]
        )
    else:
        exp_dir = args.exp_dir

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        print("Directory ", exp_dir, " Created ")
    else:
        print("Directory ", exp_dir, " already exists")

    with open(exp_dir + "/args.csv", "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=args_dict.keys())
        writer.writeheader()
        writer.writerow(args_dict)

    logger = WandbLogger(save_dir=exp_dir, project="constrained-pnns")

    trainer = Trainer(
        check_val_every_n_epoch=args.n_epochs_per_val,
        default_root_dir=exp_dir,
        fast_dev_run=args.debug,
        gpus=args.n_gpus,
        logger=logger,
        max_epochs=args.n_epochs,
    )

    trainer.fit(dynamics_model)

    trainer.test()
