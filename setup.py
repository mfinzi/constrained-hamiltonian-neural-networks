from setuptools import setup, find_packages
import sys, os

setup(
    name="hamiltonian-biases",
    description="Simplifying Hamiltonian and Lagrangian Neural Networks via Explicit Constraints",
    version="1.0",
    author="Marc Finzi and Alex Wang",
    author_email="maf820@nyu.edu",
    license="MIT",
    python_requires=">=3.6",
    install_requires=['pywavefront','networkx','wandb','pytorch_lightning',
                      'olive-oil-ml @ git+https://github.com/mfinzi/olive-oil-ml',
                      'torchdiffeq @ git+https://github.com/rtqichen/torchdiffeq',],#
    packages=find_packages(),
    long_description=open("README.md").read(),
)
