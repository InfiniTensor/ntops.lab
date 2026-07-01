import functools

import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

def arrangement(x, gamma, beta, out, hidden):
    gamma = gamma[None, :].expand((x.shape[0], -1))
    beta = beta[None, :].expand((x.shape[0], -1))
    return x.tile((1, hidden.value)), gamma.tile((1, hidden.value)), beta.tile((1, hidden.value)), out.tile((1, hidden.value)), hidden

def application(x, gamma, beta, out, hidden):
    y = x
    mean = ntl.sum(y, axis=1) / hidden
    centered = y - mean[:, None]
    var = ntl.sum(centered * centered, axis=1) / hidden
    value = centered * ntl.rsqrt(var[:, None] + 1.0e-5) * gamma + beta
    out = value * ntl.minimum(ntl.maximum(value + 3.0, 0.0), 6.0) / 6.0

@functools.cache
def _kernel(hidden):
    hidden_tensor = Tensor(0, constexpr=True, value=hidden, name="hidden")
    return ninetoothed.make(
        arrangement,
        application,
        (Tensor(2), Tensor(1), Tensor(1), Tensor(2), hidden_tensor),
        kernel_name=f"ntops_lab_layer_norm_hardswish_h{hidden}", max_num_configs=1,
    )

def run(*inputs):
    x, gamma, beta = inputs
    out = torch.empty_like(x)
    hidden = x.shape[-1]
    _kernel(hidden)(x, gamma, beta, out, hidden)
    return out
