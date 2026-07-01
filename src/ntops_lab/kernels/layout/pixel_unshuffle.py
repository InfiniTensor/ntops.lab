import torch
import ninetoothed
from ninetoothed import Tensor, block_size

BLOCK = block_size()

DOWNSCALE = 2

def arrangement(x, out):
    x_arr = x.tile((1, 1, DOWNSCALE, DOWNSCALE)).ravel().squeeze((4, 5))
    x_arr = x_arr.permute((0, 1, 4, 5, 2, 3))
    x_arr = x_arr.flatten(start_dim=1, end_dim=4)
    return x_arr.flatten().tile((BLOCK,)), out.flatten().tile((BLOCK,))

def application(x, out):
    out = x

kernel = ninetoothed.make(arrangement, application, (Tensor(4), Tensor(4)), kernel_name="ntops_lab_pixel_unshuffle")

def run(*inputs, downscale_factor=DOWNSCALE):
    (x,) = inputs
    if downscale_factor != DOWNSCALE:
        raise ValueError("pixel_unshuffle currently supports downscale_factor=2")
    n, c, h, w = x.shape
    out = torch.empty((n, c * DOWNSCALE * DOWNSCALE, h // DOWNSCALE, w // DOWNSCALE), device=x.device, dtype=x.dtype)
    kernel(x, out)
    return out
