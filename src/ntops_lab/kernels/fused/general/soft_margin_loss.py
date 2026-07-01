import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BLOCK = block_size()

def arrangement(x, target, out):
    return x.tile((BLOCK,)), target.tile((BLOCK,)), out.tile((BLOCK,))

def application(x, target, out):
    out = ntl.log(1.0 + ntl.exp(0.0 - x * target))

kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1), Tensor(1)), kernel_name="ntops_lab_soft_margin_loss")

def run(*inputs):
    x, target = inputs
    out = torch.empty_like(inputs[0])
    kernel(x, target, out)
    return out
