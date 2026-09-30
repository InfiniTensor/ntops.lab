import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

BLOCK_M = 1

def arrangement(x, out):
    return x.tile((BLOCK_M, -1)), out.tile((BLOCK_M, -1))

def application(x, out):
    m = ntl.max(x, axis=1)
    shifted = ntl.where(m[:, None] == float("-inf"), float("-inf"), x - m[:, None])
    e = ntl.exp(shifted)
    denom = ntl.sum(e, axis=1)[:, None]
    out = e / ntl.where(denom == 0.0, 1.0, denom)

kernel = ninetoothed.make(arrangement, application, (Tensor(2, other=float("-inf")), Tensor(2)), kernel_name="ntops_lab_safe_softmax")

def run(*inputs):
    x, = inputs
    out = torch.empty_like(x)
    kernel(x, out)
    return out
