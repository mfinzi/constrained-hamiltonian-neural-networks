#!/bin/bash

python trainer.py --num-epochs 50  --exp-dir "./experiments/1pendulum/NN/"        --num-masses 1 --network "NN"
python trainer.py --num-epochs 50  --exp-dir "./experiments/1pendulum/HNN/"       --num-masses 1 --network "HNN"
python trainer.py --num-epochs 50  --exp-dir "./experiments/1pendulum/LNN/"       --num-masses 1 --network "LNN"
python trainer.py --num-epochs 50  --exp-dir "./experiments/1pendulum/CHNN/"      --num-masses 1 --network "CHNN"
python trainer.py --num-epochs 50  --exp-dir "./experiments/1pendulum/CHLC/"      --num-masses 1 --network "CHLC"

python trainer.py --num-epochs 50 --exp-dir "./experiments/2pendulum/NN/"        --num-masses 2 --network "NN"
python trainer.py --num-epochs 50 --exp-dir "./experiments/2pendulum/HNN/"       --num-masses 2 --network "HNN"
python trainer.py --num-epochs 50 --exp-dir "./experiments/2pendulum/LNN/"       --num-masses 2 --network "LNN"
python trainer.py --num-epochs 50 --exp-dir "./experiments/2pendulum/CHNN/"      --num-masses 2 --network "CHNN"
python trainer.py --num-epochs 50 --exp-dir "./experiments/2pendulum/CHLC/"      --num-masses 2 --network "CHLC"

python trainer.py --num-epochs 50 --exp-dir "./experiments/3pendulum/NN/"        --num-masses 3 --network "NN"
python trainer.py --num-epochs 50 --exp-dir "./experiments/3pendulum/HNN/"       --num-masses 3 --network "HNN"
python trainer.py --num-epochs 50 --exp-dir "./experiments/3pendulum/LNN/"       --num-masses 3 --network "LNN"
python trainer.py --num-epochs 50 --exp-dir "./experiments/3pendulum/CHNN/"      --num-masses 3 --network "CHNN"
python trainer.py --num-epochs 50 --exp-dir "./experiments/3pendulum/CHLC/"      --num-masses 3 --network "CHLC"

python trainer.py --num-epochs 50 --exp-dir "./experiments/5pendulum/NN/"        --num-masses 5 --network "NN"
python trainer.py --num-epochs 50 --exp-dir "./experiments/5pendulum/HNN/"       --num-masses 5 --network "HNN"
python trainer.py --num-epochs 50 --exp-dir "./experiments/5pendulum/LNN/"       --num-masses 5 --network "LNN"
python trainer.py --num-epochs 50 --exp-dir "./experiments/5pendulum/CHNN/"      --num-masses 5 --network "CHNN"
python trainer.py --num-epochs 50 --exp-dir "./experiments/5pendulum/CHLC/"      --num-masses 5 --network "CHLC"
