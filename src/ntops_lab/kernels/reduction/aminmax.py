import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BM = 1
BN = block_size()

def arrangement(x, out0, out1):
    return x.tile((BM, BN)), out0.tile((BM,)), out1.tile((BM,))

def application(x, out0, out1):
    out0 = ntl.min(x, axis=1)
    out1 = ntl.max(x, axis=1)

kernel = ninetoothed.make(arrangement, application, (Tensor(2), Tensor(1), Tensor(1)), kernel_name="ntops_lab_aminmax")

def run(*inputs):
    x, = inputs
    out0 = torch.empty((x.shape[0],), device=x.device, dtype=x.dtype)
    out1 = torch.empty((x.shape[0],), device=x.device, dtype=x.dtype)
    kernel(x, out0, out1)
    return out0, out1
