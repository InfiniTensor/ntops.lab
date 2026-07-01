import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BLOCK_SIZE = block_size()

def arrangement(a, b, c, out):
    return a.tile((BLOCK_SIZE,)), b.tile((BLOCK_SIZE,)), c.tile((BLOCK_SIZE,)), out.tile((BLOCK_SIZE,))

def application(a, b, c, out):
    value = a + b + c
    out = ntl.minimum(ntl.maximum(value, 0.0), 6.0)

kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1), Tensor(1), Tensor(1)), kernel_name="ntops_lab_residual_bias_relu6")

def run(*inputs):
    a, b, c = inputs
    out = torch.empty_like(a)
    kernel(a, b, c, out)
    return out
