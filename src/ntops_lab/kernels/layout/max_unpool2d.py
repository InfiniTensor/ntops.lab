import functools

import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


@functools.cache
def _kernel(in_h, in_w, out_h, out_w):
    reduce = in_h * in_w

    def arrangement(x, indices, positions, out):
        x_arr = x[:, :, None, None, :, :].expand((-1, -1, out_h, out_w, -1, -1)).flatten(start_dim=4)
        idx_arr = indices[:, :, None, None, :, :].expand((-1, -1, out_h, out_w, -1, -1)).flatten(start_dim=4)
        pos_arr = positions.unsqueeze(4).expand((-1, -1, -1, -1, reduce))
        out_arr = out.unsqueeze(4).expand((-1, -1, -1, -1, reduce)).tile((1, 1, out_h, out_w, reduce))
        out_arr.dtype = out_arr.dtype.squeeze(4)
        return x_arr.tile((1, 1, out_h, out_w, reduce)), idx_arr.tile((1, 1, out_h, out_w, reduce)), pos_arr.tile((1, 1, out_h, out_w, reduce)), out_arr

    def application(x, indices, positions, out):
        out = ntl.sum(x * (indices == positions), axis=4)

    return ninetoothed.make(
        arrangement,
        application,
        (Tensor(4), Tensor(4), Tensor(4), Tensor(4)),
        kernel_name=f"ntops_lab_max_unpool2d_in{in_h}x{in_w}_out{out_h}x{out_w}",
        max_num_configs=1,
    )


def run(*inputs, kernel_size=2, stride=2, padding=0, output_size=None):
    x, indices = inputs
    if kernel_size != 2 or stride != 2 or padding != 0:
        raise ValueError("max_unpool2d currently supports kernel_size=2, stride=2, padding=0")
    if output_size is None:
        out_h, out_w = x.shape[2] * stride, x.shape[3] * stride
    else:
        out_h, out_w = int(output_size[-2]), int(output_size[-1])
    out = torch.empty((x.shape[0], x.shape[1], out_h, out_w), device=x.device, dtype=x.dtype)
    positions = torch.arange(out_h * out_w, device=x.device, dtype=indices.dtype).view(1, 1, out_h, out_w).expand(x.shape[0], x.shape[1], out_h, out_w)
    _kernel(x.shape[2], x.shape[3], out_h, out_w)(x, indices, positions, out)
    return out
