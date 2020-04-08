import math
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
import h5py
import os
import networkx as nx
from torch.utils.data import Dataset
from oil.utils.utils import Named, export, Expression, FixedNumpySeed
from biases.hamiltonian import RigidBody
from biases.chainPendulum import ChainPendulum
from biases.utils import rel_err


@export
class RigidBodyDataset(Dataset, metaclass=Named):
    space_dim = 2
    num_targets = 1

    def __init__(self,root_dir=None,body=ChainPendulum(3),n_systems=100,regen=False,
                chunk_len=5,dt=0.1,integration_time=50,angular_coords=False):
        super().__init__()
        root_dir = root_dir or os.path.expanduser(
            f"~/datasets/ODEDynamics/{self.__class__}/"
        )
        self.body = body
        filename = os.path.join(
            root_dir, f"trajectories_{body}_N{n_systems}_dt{dt}_T{integration_time}.pz"
        )
        if os.path.exists(filename) and not regen:
            ts, zs = torch.load(filename)
        else:
            ts, zs = self.generate_trajectory_data(n_systems, dt, integration_time)
            os.makedirs(root_dir, exist_ok=True)
            torch.save((ts, zs), filename)
        self.Ts, self.Zs = self.chunk_training_data(ts, zs, chunk_len)

        if angular_coords:
            N,T = self.Zs.shape[:2]
            flat_Zs = self.Zs.reshape(N*T,*self.Zs.shape[2:])
            self.Zs = self.body.global2bodyCoords(flat_Zs.double())
            print(rel_err(self.body.body2globalCoords(self.Zs.squeeze(-1)),flat_Zs))
            self.Zs = self.Zs.reshape(N,T,*self.Zs.shape[1:]).float()


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


class CartPole(RigidBody):
    def __init__(self):
        self.body_graph = nx.Graph()
        # Masses (and length) can be ignored for now since we are
        # only using the connectivity of the graph to
        # calculate DPhi, the constraint matrix
        self.body_graph.add_node(0,m=1,pos_cnstr=1) #(axis)
        self.body_graph.add_node(1,m=1)
        self.body_graph.add_edge(0,1,l=1)

class CartpoleDataset(Dataset):
    def __init__(self,root_dir=None,regen=False,batch_size=100,
                    chunk_len=5,time_limit=10,seed=0):
        super().__init__()
        self.body = CartPole()
        self.seed = seed

        root_dir = root_dir or os.path.expanduser(
            f"~/datasets/ODEDynamics/{self.__class__}/"
        )
        filename = os.path.join(
            root_dir, f"trajectories_N{batch_size}_T{time_limit}.pz"
        )
        if os.path.exists(filename) and not regen:
            ts, zs = torch.load(filename)
        else:
            ts, zs = self.generate_trajectory_data(batch_size, time_limit)
            os.makedirs(root_dir, exist_ok=True)
            torch.save((ts, zs), filename)
        self.Ts, self.Zs = self.chunk_training_data(ts, zs, chunk_len)

    def __len__(self):
        return self.Zs.shape[0]

    def __getitem__(self, i):
        return (self.Zs[i, 0], self.Ts[i]), self.Zs[i]

    def _initialize_env(self, time_limit, seed):
        from dm_control import suite
        import types

        env = suite.load(
            domain_name="cartpole", task_name="balance", task_kwargs={"random": seed}
        )

        # set time limit a la https://github.com/deepmind/dm_control/blob/03bebdf9eea0cbab480aa4882adbc4184b850835/dm_control/rl/control.py#L76
        if time_limit == float("inf"):
            env._step_limit = float("inf")
        else:
            # possible soure of error in the future if our agent does action repeats
            env._step_limit = time_limit / (env._physics.timestep() * env._n_sub_steps)

        def initialize_episode(self, physics):
            # replace https://github.com/deepmind/dm_control/blob/03bebdf9eea0cbab480aa4882adbc4184b850835/dm_control/suite/cartpole.py#L182
            physics.named.data.qpos["slider"][0] = np.array([0.0])
            physics.named.data.qpos["hinge_1"][0] = np.array([3.1415926 / 2])
            physics.named.data.qvel["slider"][0] = np.array([0.0])
            physics.named.data.qvel["hinge_1"][0] = np.array([0.0])

            # we'll use the default for Cartpole for now
            nv = physics.model.nv
            #if self._swing_up:
            #    physics.named.data.qpos["slider"] = 0.01 * self.random.randn()
            #     physics.named.data.qpos["hinge_1"] = np.pi + 0.01 * self.random.randn()
            #     physics.named.data.qpos[2:] = 0.1 * self.random.randn(nv - 2)
            # else:
            #     physics.named.data.qpos["slider"] = self.random.uniform(-0.1, 0.1)
            #     physics.named.data.qpos[1:] = self.random.uniform(-0.034, 0.034, nv - 1)
            # physics.named.data.qvel[:] = 0.01 * self.random.randn(physics.model.nv)

            # call function from super class which should be a dm_control.suite.base.Task
            # replaces `super(Balance, self).initialize_episode(physics)`
            superclass = type(env.task).mro()[1]
            superclass.initialize_episode(self, physics)
            return

        # make `initialize_episode` a bound method instead of just a static function
        # https://tryolabs.com/blog/2013/07/05/run-time-method-patching-python/
        env.task.initialize_episode = types.MethodType(initialize_episode, env.task)

        return env

    def generate_trajectory_data(self, batch_size, time_limit):
        cartesian_trajs = []
        generalized_trajs = []
        for i in range(batch_size):
            if i % 100 == 0:
                print(i)
            # new random seed each run based on `self.seed`
            env = self._initialize_env(time_limit, self.seed + i)
            cartesian_traj, generalized_traj = self._evolve(env)
            cartesian_trajs.append(cartesian_traj)
            generalized_trajs.append(generalized_traj)

        cartesian_trajs, generalized_trajs = map(
            np.stack, [cartesian_trajs, generalized_trajs]
        )
        cartesian_trajs, generalized_trajs = map(
            torch.from_numpy, [cartesian_trajs, generalized_trajs]
        )
        time = torch.linspace(
            0,
            int(env._physics.timestep() * env._n_sub_steps * env._step_limit),
            int(env._step_limit) + 1 - 1,
        )  # add one because of initial state from env.reset(), subtract one because we throw away the last state to chunk our data evenly
        time = time.view(1, -1).expand(batch_size, len(time))

        # ignore generalized_trajs for now
        return time[:,::50], cartesian_trajs[:,::50]

    def _get_com(self, env, body):
        # we only keep x and z
        #from IPython.core.debugger import set_trace; set_trace()
        return np.copy(env.physics.named.data.xipos[body][[0,2]])

    def _get_cvel(self, env, body):
        # last three entries corresponding to translational velocity
        # we only keep x and z
        return np.copy(env.physics.named.data.cvel[body][[-3,-1]])

    def _get_com_mom(self, env, body):
        return self._get_cvel(env, body) * env.physics.named.model.body_mass[body]

    def _get_q(self, env, body):
        return np.copy(env.physics.named.data.qpos[body])

    def _get_p(self, env, body):
        return np.copy(env.physics.named.data.qvel[body])

    def _evolve(self, env):
        # get the cartesian coordinate of pole and cart and their velocities
        time_step = env.reset()
        # positions
        pole_com = [self._get_com(env, "pole_1")]
        cart_com = [self._get_com(env, "cart")]
        pole_q = [self._get_q(env, "hinge_1")]
        cart_q = [self._get_q(env, "slider")]
        # momentums
        pole_com_mom = [self._get_com_mom(env, "pole_1")]
        cart_com_mom = [self._get_com_mom(env, "cart")]
        pole_p = [self._get_p(env, "hinge_1")]
        cart_p = [self._get_p(env, "slider")]

        action_spec = env.action_spec()
        while not time_step.last():
            # Take no action, just let the system evolve
            action = np.zeros(action_spec.shape)
            time_step = env.step(action)

            pole_com.append(self._get_com(env, "pole_1"))
            cart_com.append(self._get_com(env, "cart"))
            pole_q.append(self._get_q(env, "hinge_1"))
            cart_q.append(self._get_q(env, "slider"))

            pole_com_mom.append(self._get_com_mom(env, "pole_1"))
            cart_com_mom.append(self._get_com_mom(env, "cart"))
            pole_p.append(self._get_p(env, "hinge_1"))
            cart_p.append(self._get_p(env, "slider"))
        
        pole_com, pole_com_mom, cart_com, cart_com_mom = map(
            np.stack, [pole_com, pole_com_mom, cart_com, cart_com_mom]
        )
        pole_q, pole_p, cart_q, cart_p = map(np.stack, [pole_q, pole_p, cart_q, cart_p])

        com = np.stack([cart_com, pole_com])
        com_mom = np.stack([cart_com_mom, pole_com_mom])

        q = np.stack([cart_q, pole_q])
        p = np.stack([cart_p, pole_p])

        cartesian_traj = np.stack([com, com_mom])
        generalized_traj = np.stack([q, p])
        cartesian_traj = np.transpose(cartesian_traj, (2, 0, 1, 3))
        generalized_traj = np.transpose(generalized_traj, (2, 0, 1, 3))
        # output should be (number of time steps) x (q or p) x (number of bodys) x (dimension of quantity)

        # toss the final time step because we want to be able to chunk evenly
        cartesian_traj = cartesian_traj[:-1]
        generalized_traj = generalized_traj[:-1]
        return cartesian_traj, generalized_traj

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
