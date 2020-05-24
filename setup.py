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
    install_requires=['pywavefront','networkx'],#
    packages=find_packages(),
    long_description=open("README.md").read(),
)
