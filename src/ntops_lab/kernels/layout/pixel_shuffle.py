import torch
import ninetoothed
from ninetoothed import Tensor, block_size

BLOCK = block_size()

UPSCALE = 2

def arrangement(x, out):
    x_arr = x.tile((1, UPSCALE * UPSCALE, 1, 1)).ravel().squeeze((4, 6, 7))
    x_arr = x_arr.tile((1, 1, 1, 1, UPSCALE)).ravel().squeeze((5, 6, 7, 8))
    x_arr = x_arr.permute((0, 1, 2, 4, 3, 5))
    x_arr = x_arr.flatten(start_dim=2, end_dim=4).flatten(start_dim=3, end_dim=5)
    return x_arr.flatten().tile((BLOCK,)), out.flatten().tile((BLOCK,))

def application(x, out):
    out = x

kernel = ninetoothed.make(arrangement, application, (Tensor(4), Tensor(4)), kernel_name="ntops_lab_pixel_shuffle")

def run(*inputs, upscale_factor=UPSCALE):
    (x,) = inputs
    if upscale_factor != UPSCALE:
        raise ValueError("pixel_shuffle currently supports upscale_factor=2")
    n, cr2, h, w = x.shape
    channels = cr2 // (UPSCALE * UPSCALE)
    out = torch.empty((n, channels, h * UPSCALE, w * UPSCALE), device=x.device, dtype=x.dtype)
    kernel(x, out)
    return out
