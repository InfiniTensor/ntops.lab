import torch
import ninetoothed
from ninetoothed import Tensor, block_size

BLOCK = block_size()
NDIM = 2

def arrangement(x, out):
    return x.flatten().tile((BLOCK,)), out.flatten().tile((BLOCK,))

def application(x, out):
    out = x

kernel = ninetoothed.make(arrangement, application, (Tensor(NDIM), Tensor(NDIM)), kernel_name="ntops_lab_dropout")

def run(*inputs, p=0.5, training=False, inplace=False):
    (x,) = inputs
    if training and p != 0.0:
        raise ValueError("dropout currently supports training=False or p=0.0 deterministic identity only")
    if inplace:
        out = torch.empty_like(x)
        kernel(x, out)
        x.copy_(out)
        return x
    out = torch.empty_like(x)
    kernel(x, out)
    return out
