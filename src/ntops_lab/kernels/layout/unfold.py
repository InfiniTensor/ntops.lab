import torch
import ninetoothed
from ninetoothed import Tensor, block_size

KH = 3
KW = 3
PAD_H = 1
PAD_W = 1
STRIDE_H = 1
STRIDE_W = 1
BLOCK = block_size()

def arrangement(x, out):
    x_arr = x.pad(((0, 0), (0, 0), (PAD_H, PAD_H), (PAD_W, PAD_W)))
    x_arr = x_arr.tile((1, 1, KH, KW), strides=(-1, -1, STRIDE_H, STRIDE_W), floor_mode=True)
    x_arr = x_arr.ravel().squeeze((4, 5))
    x_arr = x_arr.permute((0, 1, 4, 5, 2, 3))
    x_arr = x_arr.flatten(start_dim=1, end_dim=4).flatten(start_dim=2)
    return x_arr.flatten().tile((BLOCK,)), out.flatten().tile((BLOCK,))

def application(x, out):
    out = x

kernel = ninetoothed.make(arrangement, application, (Tensor(4, other=0.0), Tensor(3)), kernel_name="ntops_lab_unfold")

def run(*inputs, kernel_size=(KH, KW), dilation=1, padding=(PAD_H, PAD_W), stride=(STRIDE_H, STRIDE_W)):
    (x,) = inputs
    if kernel_size != (KH, KW) or dilation != 1 or padding != (PAD_H, PAD_W) or stride != (STRIDE_H, STRIDE_W):
        raise ValueError("unfold currently supports kernel_size=(3,3), dilation=1, padding=(1,1), stride=(1,1)")
    n, c, h, w = x.shape
    out_h = (h + 2 * PAD_H - KH) // STRIDE_H + 1
    out_w = (w + 2 * PAD_W - KW) // STRIDE_W + 1
    out = torch.empty((n, c * KH * KW, out_h * out_w), device=x.device, dtype=x.dtype)
    kernel(x, out)
    return out
