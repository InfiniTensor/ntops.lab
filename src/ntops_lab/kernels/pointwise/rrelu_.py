import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BLOCK = block_size()

def arrangement(x, out):
    return x.tile((BLOCK,)), out.tile((BLOCK,))

def application(x, out):
    out = ntl.where(x < 0.0, x * 0.22916666666666666, x)

kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1)), kernel_name="ntops_lab_rreluinplace")

def run(*inputs):
    x, = inputs
    out = torch.empty_like(x)
    kernel(x, out)
    x.copy_(out)
    return x
