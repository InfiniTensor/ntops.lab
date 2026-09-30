import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

BLOCK_M = 1

def arrangement(x, out):
    return x.tile((BLOCK_M, -1)), out.tile((BLOCK_M,))

def application(x, out):
    value = x.to(ntl.float32)
    maximum = ntl.max(value, axis=1)
    shifted = value - maximum[:, None]
    out = maximum + ntl.log(ntl.sum(ntl.exp(shifted), axis=1))

kernel = ninetoothed.make(arrangement, application, (Tensor(2, other=float("-inf")), Tensor(1)), kernel_name="ntops_lab_logsumexp")

def run(*inputs):
    x, = inputs
    out = torch.empty((x.shape[0],), device=x.device, dtype=x.dtype)
    kernel(x, out)
    return out
