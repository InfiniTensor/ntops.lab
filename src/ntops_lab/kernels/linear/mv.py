import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BM = block_size()
BK = block_size()

def arrangement(a, x, out):
    a_arr = a.tile((BM, BK))
    x_arr = x.tile((BK,))
    x_arr = x_arr.expand((a_arr.shape[0], -1))
    out_arr = out.tile((BM,))
    return a_arr, x_arr, out_arr

def application(a, x, out):
    out = ntl.sum(a * x, axis=1)

kernel = ninetoothed.make(arrangement, application, (Tensor(2), Tensor(1), Tensor(1)), kernel_name="ntops_lab_mv")

def run(*inputs):
    a, x = inputs
    out = torch.empty((a.shape[0],), device=a.device, dtype=a.dtype)
    kernel(a, x, out)
    return out
