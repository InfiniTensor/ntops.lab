import torch
import ninetoothed
from ninetoothed import Tensor, block_size

BLOCK = block_size()

def arrangement(x, out):
    x_arr, out_arr = x.permute((2, 0, 1)).flatten(), out.flatten()
    return x_arr.tile((BLOCK,)), out_arr.tile((BLOCK,))

def application(x, out):
    out = x

kernel = ninetoothed.make(arrangement, application, (Tensor(3), Tensor(3)), kernel_name="ntops_lab_permute_3d_201")

def run(*inputs):
    x, = inputs
    out = torch.empty((x.shape[2], x.shape[0], x.shape[1]), device=x.device, dtype=x.dtype)
    kernel(x, out)
    return out
