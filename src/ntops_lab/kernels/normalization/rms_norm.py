import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BLOCK_M = 1
BLOCK_N = block_size()

def arrangement(x, weight, out):
    x_arr = x.tile((BLOCK_M, BLOCK_N))
    w_arr = weight.tile((BLOCK_N,))
    w_arr = w_arr.expand((x_arr.shape[0], -1))
    return x_arr, w_arr, out.tile((BLOCK_M, BLOCK_N))

def application(x, weight, out):
    mean_square = ntl.sum(x * x, axis=1) / 32.0
    out = x * ntl.rsqrt(mean_square[:, None] + 1.0e-5) * weight

kernel = ninetoothed.make(arrangement, application, (Tensor(2), Tensor(1), Tensor(2)), kernel_name="ntops_lab_rms_norm")

def run(*inputs):
    x, weight = inputs
    out = torch.empty_like(x)
    kernel(x, weight, out)
    return out
