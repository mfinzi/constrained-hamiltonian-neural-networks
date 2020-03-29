from setuptools import setup, find_packages
import sys, os

setup(
    name="hamiltonian-biases",
    description="",
    version="0.1",
    author="",
    author_email="maf820@nyu.edu",
    license="MIT",
    python_requires=">=3.6",
    # install_requires=['networkx','LieConv @ git+https://github.com/mfinzi/LieConv'],#
    # extras_require = {
    #    'TBX':['tensorboardX']
    # },
    packages=find_packages(),
    long_description=open("README.md").read(),
)
