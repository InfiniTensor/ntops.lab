import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BLOCK_M = 1
BLOCK_N = block_size()

def arrangement(x, weight, bias, out):
    x_arr = x.tile((BLOCK_M, BLOCK_N))
    w_arr = weight.tile((BLOCK_N,))
    w_arr = w_arr.expand((x_arr.shape[0], -1))
    b_arr = bias.tile((BLOCK_N,))
    b_arr = b_arr.expand((x_arr.shape[0], -1))
    return x_arr, w_arr, b_arr, out.tile((BLOCK_M, BLOCK_N))

def application(x, weight, bias, out):
    mean = ntl.sum(x, axis=1) / 32.0
    mean_square = ntl.sum(x * x, axis=1) / 32.0
    var = mean_square - mean * mean
    out = (x - mean[:, None]) * ntl.rsqrt(var[:, None] + 1.0e-5) * weight + bias

kernel = ninetoothed.make(arrangement, application, (Tensor(2), Tensor(1), Tensor(1), Tensor(2)), kernel_name="ntops_lab_layernorm")

def run(*inputs):
    x, weight, bias = inputs
    out = torch.empty_like(x)
    kernel(x, weight, bias, out)
    return out
