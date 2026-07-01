import torch
import ninetoothed
from ninetoothed import Tensor, block_size

BLOCK = block_size()

def arrangement(x, out):
    x_arr, out_arr = x[:, :, :].flatten(), out[:, 2:-1, :].flatten()
    return x_arr.tile((BLOCK,)), out_arr.tile((BLOCK,))

def application(x, out):
    out = x

kernel = ninetoothed.make(arrangement, application, (Tensor(3), Tensor(3)), kernel_name="ntops_lab_pad3d_middle")

def run(*inputs):
    x, = inputs
    out = torch.full((x.shape[0], x.shape[1] + 3, x.shape[2]), 0.0, device=x.device, dtype=x.dtype)
    kernel(x, out)
    return out
