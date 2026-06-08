import torch
import ninetoothed

from ninetoothed import Tensor, block_size

BLOCK_SIZE = block_size()

def arrangement(x, y, z, out):
    return x.tile((BLOCK_SIZE,)), y.tile((BLOCK_SIZE,)), z.tile((BLOCK_SIZE,)), out.tile((BLOCK_SIZE,))

def application(x, y, z, out):
    out = x + z * (y - x)

kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1), Tensor(1), Tensor(1)), kernel_name="ntops_lab_lerp")

def run(*inputs):
    x, y, z = inputs
    out = torch.empty((x.numel(),), device=x.device, dtype=x.dtype)
    kernel(x, y, z, out)
    return out
