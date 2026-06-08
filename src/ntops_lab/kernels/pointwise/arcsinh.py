import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BLOCK_SIZE = block_size()

def arrangement(x, out):
    return x.tile((BLOCK_SIZE,)), out.tile((BLOCK_SIZE,))

def application(x, out):
    out = ntl.log(x + ntl.sqrt(x * x + 1.0))

kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1)), kernel_name="ntops_lab_arcsinh")

def run(*inputs):
    x, = inputs
    out = torch.empty((x.numel(),), device=x.device, dtype=x.dtype)
    kernel(x, out)
    return out
