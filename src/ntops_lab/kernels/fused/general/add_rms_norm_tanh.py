import functools

import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

def arrangement(x, residual, weight, out, hidden):
    weight = weight[None, :].expand((x.shape[0], -1))
    return x.tile((1, hidden.value)), residual.tile((1, hidden.value)), weight.tile((1, hidden.value)), out.tile((1, hidden.value)), hidden

def application(x, residual, weight, out, hidden):
    y = x + residual
    rrms = ntl.rsqrt(ntl.sum(y * y, axis=1) / hidden + 1.0e-5)
    value = y * rrms[:, None] * weight
    out = (ntl.exp(2.0 * value) - 1.0) / (ntl.exp(2.0 * value) + 1.0)

@functools.cache
def _kernel(hidden):
    hidden_tensor = Tensor(0, constexpr=True, value=hidden, name="hidden")
    return ninetoothed.make(
        arrangement,
        application,
        (Tensor(2), Tensor(2), Tensor(1), Tensor(2), hidden_tensor),
        kernel_name=f"ntops_lab_add_rms_norm_tanh_h{hidden}", max_num_configs=1,
    )

def run(*inputs):
    x, residual, weight = inputs
    out = torch.empty_like(x)
    hidden = x.shape[-1]
    _kernel(hidden)(x, residual, weight, out, hidden)
    return out
