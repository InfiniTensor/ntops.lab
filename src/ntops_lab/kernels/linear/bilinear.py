import functools

import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size


BLOCK = block_size()
_BATCH_SHAPE_OPTIONS = ({"upper_bound": 8192}, {})


@functools.cache
def _kernel(in1, in2, out_features):
    def arrangement(x1, x2, weight, bias, out):
        x1_arr = x1[:, None, :, None].expand((-1, out_features, -1, in2))
        x2_arr = x2[:, None, None, :].expand((-1, out_features, in1, -1))
        w_arr = weight[None, :, :, :].expand((x1.shape[0], -1, -1, -1))

        x1_arr = x1_arr.flatten(end_dim=2).flatten(start_dim=1)
        x2_arr = x2_arr.flatten(end_dim=2).flatten(start_dim=1)
        w_arr = w_arr.flatten(end_dim=2).flatten(start_dim=1)

        bias_arr = bias[None, :].expand((x1.shape[0], -1)).flatten().unsqueeze(1).tile((BLOCK, -1))
        bias_arr.dtype = bias_arr.dtype.squeeze(1)
        out_arr = out.flatten().unsqueeze(1).tile((BLOCK, -1))
        out_arr.dtype = out_arr.dtype.squeeze(1)
        return x1_arr.tile((BLOCK, -1)), x2_arr.tile((BLOCK, -1)), w_arr.tile((BLOCK, -1)), bias_arr, out_arr

    def application(x1, x2, weight, bias, out):
        out = ntl.sum(x1 * x2 * weight, axis=1) + bias

    return ninetoothed.make(
        arrangement,
        application,
        (
            Tensor(shape=(None, in1), shape_options=_BATCH_SHAPE_OPTIONS),
            Tensor(shape=(None, in2), shape_options=_BATCH_SHAPE_OPTIONS),
            Tensor(shape=(out_features, in1, in2)),
            Tensor(shape=(out_features,)),
            Tensor(shape=(None, out_features), shape_options=_BATCH_SHAPE_OPTIONS),
        ),
        kernel_name=f"ntops_lab_bilinear_o{out_features}_i{in1}x{in2}",
    )


def run(*inputs):
    x1, x2, weight, bias = inputs
    out_features, in1, in2 = weight.shape
    out = torch.empty((x1.shape[0], out_features), device=x1.device, dtype=x1.dtype)
    _kernel(in1, in2, out_features)(x1, x2, weight, bias, out)
    return out
