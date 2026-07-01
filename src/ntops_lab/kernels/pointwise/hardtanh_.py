import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BLOCK = block_size()

def arrangement(x, out):
    return x.tile((BLOCK,)), out.tile((BLOCK,))

def application(x, out):
    out = ntl.minimum(ntl.maximum(x, -0.5), 0.5)

kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1)), kernel_name="ntops_lab_hardtanhinplace")

def run(*inputs):
    x, = inputs
    out = torch.empty_like(x)
    kernel(x, out)
    x.copy_(out)
    return x
