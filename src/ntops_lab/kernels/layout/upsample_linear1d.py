import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ._upsample_common import normalize_scale_factor

TILE_WIDTH = 16

def arrangement(x, out):
    xw = x.pad(((0, 0), (0, 0), (1, 1)))
    xw = xw.tile((1, 1, TILE_WIDTH + 2), strides=(-1, -1, TILE_WIDTH), floor_mode=True)
    xw = xw.ravel()
    xw = xw.flatten(end_dim=3).flatten(start_dim=1)
    xw = xw.tile((1, -1))
    xw.dtype = xw.dtype.squeeze(0)

    ow = out.tile((1, 1, TILE_WIDTH * 2), strides=(-1, -1, TILE_WIDTH * 2), floor_mode=True)
    ow = ow.ravel()
    ow = ow.flatten(end_dim=3).flatten(start_dim=1)
    ow = ow.tile((1, -1))
    ow.dtype = ow.dtype.squeeze(0)
    return xw, ow

def application(x, out):
    for i in range(16):
        prev = x[i]
        cur = x[i + 1]
        nxt = x[i + 2]
        valid = x[i + 1].offsets(-1) < x.source.shape[-1]
        first = x[i + 1].offsets(-1) == 0
        last = x[i + 1].offsets(-1) == x.source.shape[-1] - 1
        even = ntl.where(first, cur, prev * 0.25 + cur * 0.75)
        odd = ntl.where(last, cur, cur * 0.75 + nxt * 0.25)
        out[2 * i] = ntl.where(valid, even, out[2 * i])
        out[2 * i + 1] = ntl.where(valid, odd, out[2 * i + 1])

kernel = ninetoothed.make(arrangement, application, (Tensor(3, other=0.0), Tensor(3)), kernel_name="ntops_lab_upsample_linear1d_scale2_align_false", max_num_configs=1)

def run(*inputs, scale_factor=2):
    (x,) = inputs
    (scale,) = normalize_scale_factor(scale_factor, 1)
    if scale != 2:
        raise NotImplementedError("upsample_linear1d currently supports scale_factor=2")
    out = torch.empty((x.shape[0], x.shape[1], x.shape[2] * scale), device=x.device, dtype=x.dtype)
    kernel(x, out)
    return out
