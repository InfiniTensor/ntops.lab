import functools
import torch

import ninetoothed
from ninetoothed import Tensor, block_size

from ._upsample_common import normalize_scale_factor

BLOCK = block_size()

@functools.cache
def _make_kernel(scale_d, scale_h, scale_w):
    def arrangement(x, out):
        x_arr = x.unsqueeze(3).expand((-1, -1, -1, scale_d, -1, -1))
        x_arr = x_arr.unsqueeze(5).expand((-1, -1, -1, -1, -1, scale_h, -1))
        x_arr = x_arr.unsqueeze(7).expand((-1, -1, -1, -1, -1, -1, -1, scale_w))
        x_arr = x_arr.flatten(start_dim=2, end_dim=3).flatten(start_dim=3, end_dim=4).flatten(start_dim=4)
        x_arr = x_arr.flatten()
        out_arr = out.flatten()
        return x_arr.tile((BLOCK,)), out_arr.tile((BLOCK,))

    def application(x, out):
        out = x

    return ninetoothed.make(
        arrangement,
        application,
        (Tensor(5), Tensor(5)),
        kernel_name=f"ntops_lab_upsample_nearest3d_scale{scale_d}x{scale_h}x{scale_w}",
    )

def run(*inputs, scale_factor=2):
    (x,) = inputs
    scale_d, scale_h, scale_w = normalize_scale_factor(scale_factor, 3)
    out = torch.empty(
        (x.shape[0], x.shape[1], x.shape[2] * scale_d, x.shape[3] * scale_h, x.shape[4] * scale_w),
        device=x.device,
        dtype=x.dtype,
    )
    _make_kernel(scale_d, scale_h, scale_w)(x, out)
    return out
