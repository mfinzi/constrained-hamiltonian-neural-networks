import torch

def rel_err(x,y):
    return (((x-y)**2).sum()/((x+y)**2).sum()).sqrt()