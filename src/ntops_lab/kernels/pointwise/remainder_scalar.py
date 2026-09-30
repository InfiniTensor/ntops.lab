import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BLOCK_SIZE = block_size()

def arrangement(x, y, out):
    return x.tile((BLOCK_SIZE,)), y.tile((BLOCK_SIZE,)), out.tile((BLOCK_SIZE,))

def application(x, y, out):
    r = ntl.where(ntl.abs(x) < ntl.abs(1.25), x, ntl.libdevice.fmod(x, 1.25))
    out = ntl.where((r != 0.0) & ((r < 0.0) != (1.25 < 0.0)), r + 1.25, r)

kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1), Tensor(1)), kernel_name="ntops_lab_remainder_scalar")

def run(*inputs):
    x, y = inputs
    out = torch.empty((x.numel(),), device=x.device, dtype=x.dtype)
    kernel(x, y, out)
    return out
