import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BM = 1
BN = block_size()

def arrangement(x, out0, out1):
    return x.tile((BM, BN)), out0.tile((BM,)), out1.tile((BM,))

def application(x, out0, out1):
    mean = ntl.sum(x, axis=1) / 32.0
    var = ntl.sum(x * x, axis=1) / 32.0 - mean * mean
    out0 = var
    out1 = mean

kernel = ninetoothed.make(arrangement, application, (Tensor(2), Tensor(1), Tensor(1)), kernel_name="ntops_lab_var_mean")

def run(*inputs):
    x, = inputs
    out0 = torch.empty((x.shape[0],), device=x.device, dtype=x.dtype)
    out1 = torch.empty((x.shape[0],), device=x.device, dtype=x.dtype)
    kernel(x, out0, out1)
    return out0, out1
