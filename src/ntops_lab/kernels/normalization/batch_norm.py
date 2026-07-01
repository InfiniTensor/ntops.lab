import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BLOCK_M = 1
BLOCK_N = block_size()

def arrangement(x, running_mean, running_var, weight, bias, out):
    x_arr = x.tile((BLOCK_M, BLOCK_N))
    mean_arr = running_mean.tile((BLOCK_N,)).expand((x_arr.shape[0], -1))
    var_arr = running_var.tile((BLOCK_N,)).expand((x_arr.shape[0], -1))
    weight_arr = weight.tile((BLOCK_N,)).expand((x_arr.shape[0], -1))
    bias_arr = bias.tile((BLOCK_N,)).expand((x_arr.shape[0], -1))
    return x_arr, mean_arr, var_arr, weight_arr, bias_arr, out.tile((BLOCK_M, BLOCK_N))

def application(x, running_mean, running_var, weight, bias, out):
    out = (x - running_mean) * ntl.rsqrt(running_var + 1.0e-5) * weight + bias

kernel = ninetoothed.make(arrangement, application, (Tensor(2), Tensor(1), Tensor(1), Tensor(1), Tensor(1), Tensor(2)), kernel_name="ntops_lab_batch_norm")

def run(*inputs):
    x, running_mean, running_var, weight, bias = inputs
    out = torch.empty_like(x)
    kernel(x, running_mean, running_var, weight, bias, out)
    return out
