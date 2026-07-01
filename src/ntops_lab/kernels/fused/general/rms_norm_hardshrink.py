import functools

import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

def arrangement(x, weight, out, hidden):
    weight = weight[None, :].expand((x.shape[0], -1))
    return x.tile((1, hidden.value)), weight.tile((1, hidden.value)), out.tile((1, hidden.value)), hidden

def application(x, weight, out, hidden):
    value = x * ntl.rsqrt(ntl.sum(x * x, axis=1) / hidden + 1.0e-5)[:, None] * weight
    out = ntl.where(value > 0.5, value, ntl.where(value < -0.5, value, 0.0))

@functools.cache
def _kernel(hidden):
    hidden_tensor = Tensor(0, constexpr=True, value=hidden, name="hidden")
    return ninetoothed.make(
        arrangement,
        application,
        (Tensor(2), Tensor(1), Tensor(2), hidden_tensor),
        kernel_name=f"ntops_lab_rms_norm_hardshrink_h{hidden}", max_num_configs=1,
    )

def run(*inputs):
    x, weight = inputs
    out = torch.empty_like(x)
    hidden = x.shape[-1]
    _kernel(hidden)(x, weight, out, hidden)
    return out
