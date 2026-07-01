import torch
import ninetoothed
from ninetoothed import Tensor, block_size

BLOCK = block_size()

def arrangement(x, out):
    x_arr, out_arr = x[:], out[2:-3]
    return x_arr.tile((BLOCK,)), out_arr.tile((BLOCK,))

def application(x, out):
    out = x

kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1)), kernel_name="ntops_lab_pad1d_left_right")

def run(*inputs):
    x, = inputs
    out = torch.full((x.shape[0] + 5,), 0.0, device=x.device, dtype=x.dtype)
    kernel(x, out)
    return out
