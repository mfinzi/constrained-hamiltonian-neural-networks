import math
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
import h5py
import os
from torch.utils.data import Dataset
from oil.utils.utils import Named, export, Expression, FixedNumpySeed
from biases.hamiltonian import ChainPendulum


@export
class RigidBodyDataset(Dataset, metaclass=Named):
    space_dim = 2
    num_targets = 1

    def __init__(
        self,
        root_dir=None,
        body=ChainPendulum(3),
        n_systems=100,
        regen=False,
        chunk_len=5,
        dt=0.1,
        integration_time=50,
    ):
        super().__init__()
        root_dir = root_dir or os.path.expanduser(
            f"~/datasets/ODEDynamics/{self.__class__}/"
        )
        self.body = body
        filename = os.path.join(
            root_dir, f"trajectories_N{n_systems}_dt{dt}_T{integration_time}.pz"
        )
        if os.path.exists(filename) and not regen:
            ts, zs = torch.load(filename)
        else:
            ts, zs = self.generate_trajectory_data(n_systems, dt, integration_time)
            os.makedirs(root_dir, exist_ok=True)
            torch.save((ts, zs), filename)
        self.Ts, self.Zs = self.chunk_training_data(ts, zs, chunk_len)

    def __len__(self):
        return self.Zs.shape[0]

    def __getitem__(self, i):
        return (self.Zs[i, 0], self.Ts[i]), self.Zs[i]

    def generate_trajectory_data(self, n_systems, dt, integration_time, bs=100):
        """ Returns ts: (n_systems, traj_len) zs: (n_systems, traj_len, z_dim) """
        bs = min(bs, n_systems)
        n_gen = 0
        t_batches, z_batches, sysp_batches = [], [], []
        while n_gen < n_systems:
            z0s = self.sample_system(bs)
            ts = torch.arange(
                0, integration_time, dt, device=z0s.device, dtype=z0s.dtype
            )
            new_zs = self.body.integrate(z0s, ts)
            t_batches.append(ts[None].repeat(bs, 1))
            z_batches.append(new_zs)
            n_gen += bs
            print(n_gen)
        ts = torch.cat(t_batches, dim=0)[:n_systems]
        zs = torch.cat(z_batches, dim=0)[:n_systems]
        return ts, zs

    def chunk_training_data(self, ts, zs, chunk_len):
        """ Randomly samples chunks of trajectory data, returns tensors shaped for training.
        Inputs: [ts (batch_size, traj_len)] [zs (batch_size, traj_len, *z_dim)]
        outputs: [chosen_ts (batch_size, chunk_len)] [chosen_zs (batch_size, chunk_len, *z_dim)]"""
        batch_size, traj_len, *z_dim = zs.shape
        n_chunks = traj_len // chunk_len
        chunk_idx = torch.randint(0, n_chunks, (batch_size,), device=zs.device).long()
        chunked_ts = torch.stack(ts.chunk(n_chunks, dim=1))
        chunked_zs = torch.stack(zs.chunk(n_chunks, dim=1))
        chosen_ts = chunked_ts[chunk_idx, range(batch_size)]
        chosen_zs = chunked_zs[chunk_idx, torch.arange(batch_size).long()]
        return chosen_ts, chosen_zs

    def sample_system(self, N):
        """"""
        return self.body.sample_initial_conditions(N)
