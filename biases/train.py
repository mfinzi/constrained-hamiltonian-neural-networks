import copy, warnings
from oil.tuning.args import argupdated_config
from oil.datasetup.datasets import split_dataset
from oil.tuning.study import train_trial
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from oil.utils.utils import LoaderTo, islice, FixedNumpySeed, cosLr
from biases.hamiltonian import ChainPendulum
import biases.hamiltonian as hamiltonian
import biases.datasets as datasets
import biases.dynamicsTrainer as dynamicsTrainer
from biases.datasets import RigidBodyDataset
from biases.dynamicsTrainer import FC, CHFC, CHLC, IntegratedDynamicsTrainer
import lie_conv.lieGroups as lieGroups
import pickle
# from lie_conv.dynamics_trial import DynamicsTrial
try:
    import lie_conv.graphnets as graphnets
except ImportError:
    import lie_conv.lieConv as graphnets

    warnings.warn(
        "Failed to import graphnets. Please install using \
                `pip install .[GN]` for this functionality",
        ImportWarning,
    )

# network = HNN, LNN, FC, CHFC
def makeTrainer(*,network=CHFC,net_cfg={},lr=3e-3,n_train=800,regen=False,
        dataset=RigidBodyDataset,body=ChainPendulum(3),C=5,dt=0.1,
        dtype=torch.float32,angular=False,device=torch.device("cuda"),
         bs=200,num_epochs=100,trainer_config={}):
    # Create Training set and model
    splits = {"train": n_train,"test": 200}
    dataset = dataset(n_systems=1000, regen=regen, chunk_len=C,body=body,
                     dt=dt, integration_time=10,angular_coords=angular)
    with FixedNumpySeed(0):
        datasets = split_dataset(dataset, splits)
    if angular:
        model = network(G=dataset.body.body_graph,angular_dims=angular,**net_cfg).to(device=device, dtype=dtype)
    else:
        model = network(G=dataset.body.body_graph,d=2,**net_cfg).to(device=device, dtype=dtype)

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
        cfg = argupdated_config(defaults, namespace=(dynamicsTrainer, lieGroups, datasets, graphnets,hamiltonian))
        cfg.pop('local_rank')
        trainer = makeTrainer(**cfg)
        trainer.train(cfg['num_epochs'])
        rollouts = trainer.test_rollouts(angular_to_euclidean=cfg['angular'])
        fname = f"rollout_errs_{cfg['network']}_{cfg['body']}.np".replace('(','').replace(')','')
        with open(fname,'wb') as f:
            pickle.dump(rollouts,f)
        #defaults["trainer_config"]["early_stop_metric"] = "val_MSE"
        #print(Trial()))
