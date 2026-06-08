import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BLOCK_N = block_size()

def arrangement(x, y, out):
    return x.tile((BLOCK_N,)), y.tile((BLOCK_N,)), out.tile((1,))

def application(x, y, out):
    out = ntl.sum(x * y)

kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1), Tensor(1)), kernel_name="ntops_lab_dot")

def run(*inputs):
    x, y = inputs
    out = torch.empty((1,), device=x.device, dtype=x.dtype)
    kernel(x, y, out)
    return out
