
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
from .utils import Named, export, Expression, FixedNumpySeed, RandomZrotation, GaussianNoise
from oil.datasetup.datasets import EasyIMGDataset
from lie_conv.hamiltonian import HamiltonianDynamics, KeplerH, SpringH
from lie_conv.lieGroups import SO3
from torchdiffeq import odeint_adjoint as odeint


class DynamicsDataset(Dataset, metaclass=Named):
    num_targets = 1

    def __len__(self):
        return self.Zs.shape[0]

    def __getitem__(self, i):
        return (self.Zs[i, 0], self.Ts[i]), self.Zs[i]

    def generate_trajectory_data(self, n_systems, sim_kwargs, batch_size=5000):
        """ Returns ts: (n_systems, traj_len) zs: (n_systems, traj_len, z_dim) """
        batch_size = min(batch_size, n_systems)
        n_gen = 0
        t_batches, z_batches, sysp_batches = [], [], []
        while n_gen < n_systems:
            z0s = self.sample_system(n_systems=batch_size, space_dim=self.space_dim)
            new_ts, new_zs = self.integrate(z0s,**sim_kwargs)
            t_batches.append(new_ts)
            z_batches.append(new_zs)
            n_gen += new_ts.shape[0]
        ts = torch.cat(t_batches, dim=0)[:n_systems]
        zs = torch.cat(z_batches, dim=0)[:n_systems]
        return ts, zs

    def integrate(self, z0, traj_len, delta_t):
        with torch.no_grad():
            ts = torch.linspace(0, traj_len * delta_t, traj_len).double()[None].expand(bs,-1)
            zs = odeint(self.dynamics, z0, ts, rtol=1e-8, method='rk4').detach()
        return ts, zs

    def chunk_training_data(self, ts, zs, chunk_len):
        """ Randomly samples chunks of trajectory data, returns tensors shaped for training.
        Inputs: [ts (batch_size, traj_len)] [zs (batch_size, traj_len, z_dim)]
        outputs: [chosen_ts (batch_size, chunk_len)] [chosen_zs (batch_size, chunk_len, z_dim)]"""
        batch_size, traj_len, z_dim = zs.shape
        n_chunks = traj_len // chunk_len
        chunk_idx = torch.randint(0, n_chunks, (batch_size,), device=zs.device).long()
        chunked_ts = torch.stack(ts.chunk(n_chunks, dim=1))
        chunked_zs = torch.stack(zs.chunk(n_chunks, dim=1))
        chosen_ts = chunked_ts[chunk_idx, range(batch_size)]
        chosen_zs = chunked_zs[chunk_idx, torch.arange(batch_size).long()]
        return chosen_ts, chosen_zs

    def sample_system(self, n_systems, space_dim, **kwargs):
        """output: [z0 (n_systems, z_dim)] """
        raise NotImplementedError


@export
class SpringDynamics(DynamicsDataset):
    default_root_dir = os.path.expanduser('~/datasets/ODEDynamics/SpringDynamics/')
    sys_dim = 2
    
    def __init__(self, root_dir=default_root_dir, train=True, download=True, n_systems=100, space_dim=2, regen=False,
                 chunk_len=5):
        super().__init__()
        filename = os.path.join(root_dir, f"spring_{space_dim}D_{n_systems}_{('train' if train else 'test')}.pz")
        self.space_dim = space_dim
        if os.path.exists(filename) and not regen:
            ts, zs = torch.load(filename)
        elif download:
            sim_kwargs = dict(
                traj_len=500,
                delta_t=0.01,
            )
            ts, zs, self.SysP = self.generate_trajectory_data(n_systems=n_systems, sim_kwargs=sim_kwargs)
            os.makedirs(root_dir, exist_ok=True)
            print(filename)
            torch.save((ts, zs),filename)
        else:
            raise Exception("Download=False and data not there")
        self.Ts, self.Zs = self.chunk_training_data(ts, zs, chunk_len)
    
    def sample_system(self, n_systems, space_dim, ood=False):
        """"""
        n = np.random.choice([6]) #TODO: handle padding/batching with different n
        if ood: n = np.random.choice([4,8])
        masses = (3 * torch.rand(n_systems, n).double() + .1)
        k = 5*torch.rand(n_systems, n).double()
        q0 = .4*torch.randn(n_systems, n, space_dim).double()
        p0 = .6*torch.randn(n_systems, n, space_dim).double()
        p0 -= p0.mean(0,keepdim=True)
        z0 = torch.cat([q0.reshape(n_systems, n * space_dim), p0.reshape(n_systems, n * space_dim)], dim=1)
        return z0, (masses, k)

    def dynamics(self, sys_params):
        H = lambda t, z: SpringH(z, *sys_params)
        return HamiltonianDynamics(H, wgrad=False)