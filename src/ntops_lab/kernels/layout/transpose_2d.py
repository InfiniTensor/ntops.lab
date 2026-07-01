import torch
import ninetoothed
from ninetoothed import Tensor, block_size

BLOCK = block_size()

def arrangement(x, out):
    x_arr, out_arr = x.permute((1, 0)).flatten(), out.flatten()
    return x_arr.tile((BLOCK,)), out_arr.tile((BLOCK,))

def application(x, out):
    out = x

kernel = ninetoothed.make(arrangement, application, (Tensor(2), Tensor(2)), kernel_name="ntops_lab_transpose_2d")

def run(*inputs):
    x, = inputs
    out = torch.empty((x.shape[1], x.shape[0]), device=x.device, dtype=x.dtype)
    kernel(x, out)
    return out
