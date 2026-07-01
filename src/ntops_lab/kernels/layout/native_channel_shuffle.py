import torch
import ninetoothed
from ninetoothed import Tensor, block_size

BLOCK = block_size()

GROUPS = 4

def arrangement(x, out):
    channels_per_group = x.shape[1] // GROUPS
    x_arr = x.tile((1, channels_per_group, 1, 1)).ravel().squeeze((4, 6, 7))
    x_arr = x_arr.permute((0, 4, 1, 2, 3))
    x_arr = x_arr.flatten(start_dim=1, end_dim=3)
    return x_arr.flatten().tile((BLOCK,)), out.flatten().tile((BLOCK,))

def application(x, out):
    out = x

kernel = ninetoothed.make(arrangement, application, (Tensor(4), Tensor(4)), kernel_name="ntops_lab_native_channel_shuffle")

def run(*inputs, groups=GROUPS):
    (x,) = inputs
    if groups != GROUPS:
        raise ValueError("channel_shuffle currently supports groups=4")
    out = torch.empty_like(x)
    kernel(x, out)
    return out
