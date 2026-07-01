import torch
import ninetoothed
from ninetoothed import Tensor, block_size

BLOCK = block_size()

def arrangement(x, out):
    x_arr, out_arr = x.flatten(start_dim=1).flatten(), out.flatten()
    return x_arr.tile((BLOCK,)), out_arr.tile((BLOCK,))

def application(x, out):
    out = x

kernel = ninetoothed.make(arrangement, application, (Tensor(3), Tensor(2)), kernel_name="ntops_lab_flatten_dim1_3d")

def run(*inputs):
    x, = inputs
    out = torch.empty((x.shape[0], x.shape[1] * x.shape[2]), device=x.device, dtype=x.dtype)
    kernel(x, out)
    return out
