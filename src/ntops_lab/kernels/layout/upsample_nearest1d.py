import functools
import torch

import ninetoothed
from ninetoothed import Tensor, block_size

from ._upsample_common import normalize_scale_factor

BLOCK = block_size()

@functools.cache
def _make_kernel(scale):
    def arrangement(x, out):
        x_arr = x.unsqueeze(3).expand((-1, -1, -1, scale)).flatten(start_dim=2)
        x_arr = x_arr.flatten()
        out_arr = out.flatten()
        return x_arr.tile((BLOCK,)), out_arr.tile((BLOCK,))

    def application(x, out):
        out = x

    return ninetoothed.make(
        arrangement,
        application,
        (Tensor(3), Tensor(3)),
        kernel_name=f"ntops_lab_upsample_nearest1d_scale{scale}",
    )

def run(*inputs, scale_factor=2):
    (x,) = inputs
    (scale,) = normalize_scale_factor(scale_factor, 1)
    out = torch.empty((x.shape[0], x.shape[1], x.shape[2] * scale), device=x.device, dtype=x.dtype)
    _make_kernel(scale)(x, out)
    return out
