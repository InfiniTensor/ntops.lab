import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BLOCK_SIZE = block_size()

def arrangement(x, out):
    return x.tile((BLOCK_SIZE,)), out.tile((BLOCK_SIZE,))

def application(x, out):
    lower = ntl.floor(x)
    fraction = x - lower
    odd = lower - 2.0 * ntl.floor(lower * 0.5)
    out = ntl.where(fraction > 0.5, lower + 1.0, ntl.where(fraction == 0.5, lower + odd, lower))

kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1)), kernel_name="ntops_lab_round_out")

def run(*inputs):
    x, = inputs
    out = torch.empty((x.numel(),), device=x.device, dtype=x.dtype)
    kernel(x, out)
    return out
