import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

BLOCK_M = 1

def arrangement(x, out):
    return x.tile((BLOCK_M, -1)), out.tile((BLOCK_M, -1))

def application(x, out):
    value = x.to(ntl.float32)
    m = ntl.max(value, axis=1)
    e = ntl.exp(value - m[:, None])
    out = e / ntl.sum(e, axis=1)[:, None]

kernel = ninetoothed.make(arrangement, application, (Tensor(2, other=float("-inf")), Tensor(2)), kernel_name="ntops_lab_scaled_softmax")

def run(*inputs):
    x, = inputs
    out = torch.empty_like(x)
    kernel(x, out)
    return out
