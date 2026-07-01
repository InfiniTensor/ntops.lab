import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BLOCK_SIZE = block_size()

def arrangement(a, b, out):
    return a.tile((BLOCK_SIZE,)), b.tile((BLOCK_SIZE,)), out.tile((BLOCK_SIZE,))

def application(a, b, out):
    out = (a + b) * ((ntl.exp(2.0 * ntl.log(1.0 + ntl.exp((a + b)))) - 1.0) / (ntl.exp(2.0 * ntl.log(1.0 + ntl.exp((a + b)))) + 1.0))

kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1), Tensor(1)), kernel_name="ntops_lab_residual_mish")

def run(*inputs):
    a, b = inputs
    out = torch.empty_like(a)
    kernel(a, b, out)
    return out
