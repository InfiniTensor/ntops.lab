import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BLOCK = block_size()

def arrangement(x, target, var, out):
    return x.tile((BLOCK,)), target.tile((BLOCK,)), var.tile((BLOCK,)), out.tile((BLOCK,))

def application(x, target, var, out):
    diff = x - target
    vc = ntl.maximum(var, 1.0e-6)
    out = 0.5 * (ntl.log(vc) + diff * diff / vc)

kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1), Tensor(1), Tensor(1)), kernel_name="ntops_lab_gaussian_nll_loss")

def run(*inputs):
    x, target, var = inputs
    out = torch.empty_like(inputs[0])
    kernel(x, target, var, out)
    return out
