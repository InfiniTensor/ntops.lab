import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BLOCK_SIZE = block_size()

def arrangement(a, b, out):
    return a.tile((BLOCK_SIZE,)), b.tile((BLOCK_SIZE,)), out.tile((BLOCK_SIZE,))

def application(a, b, out):
    value = a
    out = (ntl.where(value > 0.0, value, ntl.exp(value) - 1.0)) * b

kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1), Tensor(1)), kernel_name="ntops_lab_gate_elu_mul")

def run(*inputs):
    a, b = inputs
    out = torch.empty_like(a)
    kernel(a, b, out)
    return out
