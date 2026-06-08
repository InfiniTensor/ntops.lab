import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BM = block_size()
BK = block_size()

def arrangement(bias, a, x, out):
    a_arr = a.tile((BM, BK))
    x_arr = x.tile((BK,))
    x_arr = x_arr.expand((a_arr.shape[0], -1))
    out_arr = out.tile((BM,))
    bias_arr = bias.tile((BM,))
    return bias_arr, a_arr, x_arr, out_arr

def application(bias, a, x, out):
    out = bias + ntl.sum(a * x, axis=1)

kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(2), Tensor(1), Tensor(1)), kernel_name="ntops_lab_addmv")

def run(*inputs):
    bias, a, x = inputs
    out = torch.empty((a.shape[0],), device=a.device, dtype=a.dtype)
    kernel(bias, a, x, out)
    return out
