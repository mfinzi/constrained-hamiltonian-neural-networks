# Simplifying Hamiltonian and Lagrangian Neural Networks via Explicit Constraints

# Appendix
The appendix is in this directory as `Appendix.pdf`


# Code
Our code in the `biases` directory relies on some publically available codebases which we package together
as a conda environment.

## Installation instructions
1. Install anaconda or miniconda
2. Create a conda environment with `conda env create -f conda_env.yml`
3. Install our codebase with `pip install ./` while in the current directory
4. Create a wandb account for experiment tracking

## Example usage

```
python pl_trainer.py --network-class CHNN --body-class Gyroscope --wandb-project "YOUR WANDB PROJECT"
python pl_trainer.py --network-class CLNN --body-class Gyroscope --wandb-project "YOUR WANDB PROJECT"
python pl_trainer.py --network-class HNN --body-class Gyroscope --wandb-project "YOUR WANDB PROJECT"
python pl_trainer.py --network-class DeLaN --body-class Gyroscope --wandb-project "YOUR WANDB PROJECT"
python pl_trainer.py --network-class NN --body-class Gyroscope --wandb-project "YOUR WANDB PROJECT"
```

Note that here `NN` corresponds to a Neural ODE.



