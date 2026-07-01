import functools

import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(x, target, out, dim):
    return x.tile((1, dim.value)), target.tile((1, dim.value)), out.tile((1,)), dim


def application(x, target, out, dim):
    abs_x = ntl.where(x < 0.0, 0.0 - x, x)
    loss = ntl.maximum(x, 0.0) - x * target + ntl.log(1.0 + ntl.exp(0.0 - abs_x))
    out = ntl.sum(loss, axis=1) / dim


@functools.cache
def _kernel(dim):
    dim_tensor = Tensor(0, constexpr=True, value=dim, name="dim")
    return ninetoothed.make(
        arrangement,
        application,
        (Tensor(2), Tensor(2), Tensor(1), dim_tensor),
        kernel_name=f"ntops_lab_multilabel_soft_margin_loss_d{dim}",
        max_num_configs=1,
    )


def run(*inputs):
    x, target = inputs
    out = torch.empty((x.shape[0],), device=x.device, dtype=x.dtype)
    dim = x.shape[-1]
    _kernel(dim)(x, target, out, dim)
    return out
