import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BLOCK = block_size()

def arrangement(x, target, out):
    return x.tile((BLOCK,)), target.tile((BLOCK,)), out.tile((BLOCK,))

def application(x, target, out):
    eps = 1.0e-7
    xc = ntl.minimum(ntl.maximum(x, eps), 1.0 - eps)
    out = 0.0 - (target * ntl.log(xc) + (1.0 - target) * ntl.log(1.0 - xc))

kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1), Tensor(1)), kernel_name="ntops_lab_binary_cross_entropy")

def run(*inputs):
    x, target = inputs
    out = torch.empty_like(inputs[0])
    kernel(x, target, out)
    return out
