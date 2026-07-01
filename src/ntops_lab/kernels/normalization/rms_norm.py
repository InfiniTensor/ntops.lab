import functools

import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BLOCK_M = 1
BLOCK_N = block_size()

def arrangement(x, weight, out, hidden):
    x_arr = x.tile((BLOCK_M, BLOCK_N))
    w_arr = weight.tile((BLOCK_N,))
    w_arr = w_arr.expand((x_arr.shape[0], -1))
    return x_arr, w_arr, out.tile((BLOCK_M, BLOCK_N)), hidden

def application(x, weight, out, hidden):
    mean_square = ntl.sum(x * x, axis=1) / hidden
    out = x * ntl.rsqrt(mean_square[:, None] + 1.0e-5) * weight

@functools.cache
def _kernel(hidden):
    hidden_tensor = Tensor(0, constexpr=True, value=hidden, name="hidden")
    return ninetoothed.make(
        arrangement,
        application,
        (Tensor(2), Tensor(1), Tensor(2), hidden_tensor),
        kernel_name=f"ntops_lab_rms_norm_h{hidden}",
    )

def run(*inputs):
    x, weight = inputs
    out = torch.empty_like(x)
    hidden = x.shape[-1]
    _kernel(hidden)(x, weight, out, hidden)
    return out
