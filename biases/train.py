import copy, warnings
from oil.tuning.args import argupdated_config
from oil.datasetup.datasets import split_dataset
from oil.tuning.study import train_trial
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from oil.utils.utils import LoaderTo, islice, FixedNumpySeed, cosLr
from biases.systems.chain_pendulum import ChainPendulum
import biases.datasets as datasets
from biases.models import HNN,LNN,NN,CHNN,CHLC,CH
from biases.datasets import RigidBodyDataset
from biases.dynamics_trainer import IntegratedDynamicsTrainer
import biases.models as models
import biases.systems as systems
import lie_conv.lieGroups as lieGroups
import pickle

# network = HNN, LNN, NN, CHNN
def makeTrainer(*,network=CHNN,net_cfg={},lr=3e-3,n_train=800,regen=False,
        dataset=RigidBodyDataset,body=ChainPendulum(3),C=5,dt=0.1,
        dtype=torch.float32,device=torch.device("cuda"),
        bs=200,num_epochs=100,trainer_config={}):
    # Create Training set and model
    angular = not issubclass(network,CH)
    splits = {"train": n_train,"test": 200}
    with FixedNumpySeed(0):
        dataset = dataset(n_systems=1000, regen=regen, chunk_len=C,body=body,
                        dt=dt, integration_time=10,angular_coords=angular)
        datasets = split_dataset(dataset, splits)

    
    dof_ndim = dataset.body.D if angular else dataset.body.d
    model = network(dataset.body.body_graph,dof_ndim =dof_ndim,
                    angular_dims=dataset.body.angular_dims,**net_cfg)
    model = model.to(device=device, dtype=dtype)
    # Create train and Dev(Test) dataloaders and move elems to gpu
    dataloaders = {k: LoaderTo(
                DataLoader(v, batch_size=min(bs, splits[k]), num_workers=0, shuffle=(k == "train")),
                device=device,dtype=dtype) for k, v in datasets.items()}
    dataloaders["Train"] = dataloaders["train"]
    # Initialize optimizer and learning rate schedule
    opt_constr = lambda params: Adam(params, lr=lr)
    lr_sched = cosLr(num_epochs)
    return IntegratedDynamicsTrainer(model,dataloaders,opt_constr,lr_sched,
                            log_args={"timeFrac": 1 / 4, "minPeriod": 0.0},**trainer_config)


#Trial = train_trial(makeTrainer)
if __name__ == "__main__":
    with FixedNumpySeed(0):
        defaults = copy.deepcopy(makeTrainer.__kwdefaults__)
        #defaults["save"] = False
        namespace = (lieGroups,datasets,systems,models)
        cfg = argupdated_config(defaults, namespace=namespace)
        cfg.pop('local_rank')
        trainer = makeTrainer(**cfg)
        trainer.train(cfg['num_epochs'])
        rollouts = trainer.test_rollouts(angular_to_euclidean= not issubclass(cfg['network'],CH))
        fname = f"rollout_errs_{cfg['network']}_{cfg['body']}.np"
        with open(fname,'wb') as f:
            pickle.dump(rollouts,f)
        #defaults["trainer_config"]["early_stop_metric"] = "val_MSE"
        #print(Trial()))
