[![Code Climate maintainability](https://api.codeclimate.com/v1/badges/a99a88d28ad37a79dbf6/maintainability)](https://codeclimate.com/github/mfinzi/hamiltonian-biases/maintainability)
# Simplifying Hamiltonian and Lagrangian Neural Networks via Explicit Constraints
<p align="center">
  <img src="https://user-images.githubusercontent.com/12687085/94082856-da416d80-fdcf-11ea-8d69-bad3c604c35e.png" width=900>
</p>
This repo contains the implementation and the experiments for the paper 

[Simplifying Hamiltonian and Lagrangian Neural Networks via Explicit Constraints](https://arxiv.org/abs/2002.12880)
by [Marc Finzi](https://mfinzi.github.io/), [Alex Wang](https://keawang.github.io/), and [Andrew Gordon Wilson](https://cims.nyu.edu/~andrewgw/). 


<!-- ![CHNN_perf_summary](https://user-images.githubusercontent.com/12687085/94081992-e75d5d00-fdcd-11ea-9df0-576af6909944.PNG) -->
<!-- ![chaotic_2pendulum](https://user-images.githubusercontent.com/12687085/94081997-e9bfb700-fdcd-11ea-8ca1-ce7ce1cdc717.PNG) -->
<!-- ![systems](https://user-images.githubusercontent.com/12687085/94081999-eb897a80-fdcd-11ea-8e29-c676d4e25f64.PNG) -->

# Code
Our code in the `biases` directory relies on some publically available codebases which we package together
as a conda environment.

# Installation instructions
```bash
git clone https://github.com/mfinzi/hamiltonian-biases.git
cd hamiltonian-biases
pip install -e .
```
(Optional) Create a wandb account for experiment tracking

# Train Models
We have implemented a variety of challenging benchmarks for modeling physical dynamical systems such as ``ChainPendulum``, ``CoupledPendulum``,``MagnetPendulum``,``Gyroscope``,``Rotor`` which can be selected with the ``--body-class`` argument.

<p align="center">
  <img src="https://user-images.githubusercontent.com/12687085/94081999-eb897a80-fdcd-11ea-8e29-c676d4e25f64.PNG" width=1000>
</p>

You can run our models ``CHNN`` and ``CLNN`` as well as the baseline ``NN`` (NeuralODE), ``DeLaN``, and ``HNN`` models with the ``network-class`` argument as shown below.

```
python pl_trainer.py --network-class CHNN --body-class Gyroscope --wandb-project "YOUR WANDB PROJECT"
python pl_trainer.py --network-class CLNN --body-class Gyroscope --wandb-project "YOUR WANDB PROJECT"
python pl_trainer.py --network-class HNN --body-class Gyroscope --wandb-project "YOUR WANDB PROJECT"
python pl_trainer.py --network-class DeLaN --body-class Gyroscope --wandb-project "YOUR WANDB PROJECT"
python pl_trainer.py --network-class NN --body-class Gyroscope --wandb-project "YOUR WANDB PROJECT"
```

Our explicitly constrained ``CHNN`` and ``CLNN`` outperform the competing methods by several orders of magnitude across the different benchmarks as shown below.
<p align="center">
  <img src="https://user-images.githubusercontent.com/12687085/94081992-e75d5d00-fdcd-11ea-9df0-576af6909944.PNG" width=1000>
</p>

If you find our work helpful, please cite it with
```bibtex
@article{finzi2020simplifying,
  title={Simplifying Hamiltonian and Lagrangian Neural Networks via Explicit Constraints},
  author={Finzi, Marc and Wang, Alex and Wilson, Andrew Gordon},
  journal={NeurIPS},
  year={2020}
}
```

