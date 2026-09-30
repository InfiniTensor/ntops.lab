import functools

import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

BLOCK_M = 1

def arrangement(x, weight, bias, out, hidden):
    x_arr = x.tile((BLOCK_M, hidden.value))
    w_arr = weight.tile((hidden.value,))
    w_arr = w_arr.expand((x_arr.shape[0], -1))
    b_arr = bias.tile((hidden.value,))
    b_arr = b_arr.expand((x_arr.shape[0], -1))
    return x_arr, w_arr, b_arr, out.tile((BLOCK_M, hidden.value)), hidden

def application(x, weight, bias, out, hidden):
    value = x.to(ntl.float32)
    mean = ntl.sum(value, axis=1) / hidden
    centered = value - mean[:, None]
    var = ntl.sum(centered * centered, axis=1) / hidden
    out = (value - mean[:, None]) * ntl.rsqrt(var[:, None] + 1.0e-5) * weight + bias

@functools.cache
def _kernel(hidden):
    hidden_tensor = Tensor(0, constexpr=True, value=hidden, name="hidden")
    return ninetoothed.make(
        arrangement,
        application,
        (Tensor(2), Tensor(1), Tensor(1), Tensor(2), hidden_tensor),
        kernel_name=f"ntops_lab_layer_norm_h{hidden}",
    )

def run(*inputs):
    x, weight, bias = inputs
    out = torch.empty_like(x)
    hidden = x.shape[-1]
    _kernel(hidden)(x, weight, bias, out, hidden)
    return out
