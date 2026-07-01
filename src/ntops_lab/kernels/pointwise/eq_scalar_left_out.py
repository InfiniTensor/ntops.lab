import torch
import ninetoothed
from ninetoothed import Tensor, block_size

BLOCK_SIZE = block_size()

def arrangement(x, y, out):
    return x.tile((BLOCK_SIZE,)), y.tile((BLOCK_SIZE,)), out.tile((BLOCK_SIZE,))

def application(x, y, out):
    out = 0.25 == x

kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1), Tensor(1)), kernel_name="ntops_lab_eq_scalar_left_out")

def run(*inputs):
    x, y = inputs
    out = torch.empty((x.numel(),), device=x.device, dtype=torch.bool)
    kernel(x, y, out)
    return out
