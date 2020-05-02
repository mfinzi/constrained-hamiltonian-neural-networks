from oil.datasetup.datasets import split_dataset
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from oil.utils.utils import LoaderTo, FixedNumpySeed, cosLr
from biases.datasets import RigidBodyDataset
from biases.dynamics_trainer import IntegratedDynamicsTrainer
from biases.models.constrained_hnn import CHNN, CHLC
from biases.models.hnn import HNN
from biases.models.lnn import LNN
from biases.models.nn import NN, DeltaNN
from biases.systems.chain_pendulum import ChainPendulum
from typing import Union, Tuple
import sys
import argparse
import numpy as np
import csv
import os


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


def make_trainer(
    chunk_len: int,
    angular: Union[Tuple, bool],
    body,
    bs: int,
    dataset,
    dt: float,
    lr: float,
    n_train: int,
    n_val: int,
    n_test: int,
    net_cfg: dict,
    network,
    num_epochs: int,
    regen: bool,
    seed: int = 0,
    device=torch.device("cuda"),
    dtype=torch.float32,
    trainer_config={},
):
    # Create Training set and model
    splits = {"train": n_train, "val": n_val, "test": n_test}
    dataset = dataset(
        n_systems=n_train + n_val + n_test,
        regen=regen,
        chunk_len=chunk_len,
        body=body,
        dt=dt,
        integration_time=10,
        angular_coords=angular,
    )
    # dataset=CartpoleDataset(batch_size=500,regen=regen)
    with FixedNumpySeed(seed):
        datasets = split_dataset(dataset, splits)
    model = network(G=dataset.body.body_graph, **net_cfg).to(device=device, dtype=dtype)

    # Create train and Dev(Test) dataloaders and move elems to gpu
    dataloaders = {
        k: LoaderTo(
            DataLoader(
                v, batch_size=min(bs, splits[k]), num_workers=0, shuffle=(k == "train")
            ),
            device=device,
            dtype=dtype,
        )
        for k, v in datasets.items()
    }
    dataloaders["Train"] = dataloaders["train"]
    # Initialize optimizer and learning rate schedule
    opt_constr = lambda params: Adam(params, lr=lr)
    lr_sched = cosLr(num_epochs)
    return IntegratedDynamicsTrainer(
        model,
        dataloaders,
        opt_constr,
        lr_sched,
        log_args={"timeFrac": 1 / 4, "minPeriod": 0.0},
        **trainer_config
    )


def parse_cmdline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=200, help="Batch size")
    parser.add_argument(
        "--chunk-len",
        type=int,
        default=5,
        help="Length of each chunk of training trajectory",
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="./",
        help="Directory to save files from this experiment",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=200, help="Number of hidden units"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
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
        "--network",
        type=str,
        help="Dynamics network",
        choices=["NN", "DeltaNN", "HNN", "LNN", "CHNN", "CHLC"],
    )
    parser.add_argument(
        "--num-epochs", type=int, default=300, help="Number of training epochs"
    )
    parser.add_argument(
        "--num-layers", type=int, default=3, help="Number of hidden layers"
    )
    parser.add_argument(
        "--num-masses", type=int, default=1, help="Number of masses in ChainPendulum",
    )
    parser.add_argument(
        "--regen",
        action="store_true",
        default=False,
        help="Forcibly regenerate training data",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cmdline()

    body = ChainPendulum(args.num_masses)

    euclidean_coords = args.network not in [
        "NN",
        "LNN",
        "HNN",
    ]  # TODO: try NN in euclidean

    net_cfg = {
        "dof_ndim": 2 if euclidean_coords else 1,  # 2 because we are only doing pendulums for now
        "angular_dims": tuple() if euclidean_coords else True,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "wgrad": True,
    }

    trainer = make_trainer(
        angular=not euclidean_coords,
        body=body,
        bs=args.batch_size,
        chunk_len=args.chunk_len,
        dataset=RigidBodyDataset,
        dt=0.1,
        lr=args.lr,
        network=str_to_class(args.network),
        n_test=args.n_test,
        n_train=args.n_train,
        n_val=args.n_val,
        net_cfg=net_cfg,
        num_epochs=args.num_epochs,
        regen=args.regen,
    )

    # Create target directory & all intermediate directories if don't exists
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
        print("Directory ", args.exp_dir, " Created ")
    else:
        print("Directory ", args.exp_dir, " already exists")

    with open(args.exp_dir + "/args.csv", "w") as csvfile:
        args_dict = vars(args)  # convert to dict
        writer = csv.DictWriter(csvfile, fieldnames=args_dict.keys())
        writer.writeheader()
        writer.writerow(args_dict)

    trainer.train(args.num_epochs)

    print("Saving training logs")
    ax = trainer.logger.scalar_frame.plot()
    ax.set(yscale="log")
    figure_path = args.exp_dir + "/log.png"
    ax.figure.savefig(figure_path)

    print("Saving test rollouts")
    rollouts_path = args.exp_dir + "/test_rollouts"
    rollout_errs = trainer.test_rollouts(
        angular_to_euclidean=not euclidean_coords, pert_eps=1e-4
    )
    np.save(rollouts_path, rollout_errs.detach().cpu().numpy())

    print("Saving model state_dict")
    model_path = args.exp_dir + "/model.pt"
    torch.save(trainer.model.to("cpu").state_dict(), model_path)
