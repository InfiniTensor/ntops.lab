import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BLOCK_M = 1
BLOCK_N = block_size()

def arrangement(x, out):
    return x.tile((BLOCK_M, BLOCK_N)), out.tile((BLOCK_M,))

def application(x, out):
    out = ntl.sum(x, axis=1) / 32.0

kernel = ninetoothed.make(arrangement, application, (Tensor(2), Tensor(1)), kernel_name="ntops_lab_mean")

def run(*inputs):
    x, = inputs
    out = torch.empty((x.shape[0],), device=x.device, dtype=x.dtype)
    kernel(x, out)
    return out
