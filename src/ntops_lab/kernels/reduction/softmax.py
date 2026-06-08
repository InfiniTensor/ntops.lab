import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BLOCK_M = 1
BLOCK_N = block_size()

def arrangement(x, out):
    return x.tile((BLOCK_M, BLOCK_N)), out.tile((BLOCK_M, BLOCK_N))

def application(x, out):
    m = ntl.max(x, axis=1)
    e = ntl.exp(x - m[:, None])
    out = e / ntl.sum(e, axis=1)[:, None]

kernel = ninetoothed.make(arrangement, application, (Tensor(2), Tensor(2)), kernel_name="ntops_lab_softmax")

def run(*inputs):
    x, = inputs
    out = torch.empty_like(x)
    kernel(x, out)
    return out
