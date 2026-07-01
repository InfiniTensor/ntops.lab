import functools

import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(x, out0, out1, dim):
    return x.tile((1, dim.value)), out0.tile((1,)), out1.tile((1,)), dim


def application(x, out0, out1, dim):
    mean = ntl.sum(x, axis=1) / dim
    var = ntl.sum(x * x, axis=1) / dim - mean * mean
    out0 = var
    out1 = mean


@functools.cache
def _kernel(dim):
    dim_tensor = Tensor(0, constexpr=True, value=dim, name="dim")
    return ninetoothed.make(
        arrangement,
        application,
        (Tensor(2), Tensor(1), Tensor(1), dim_tensor),
        kernel_name=f"ntops_lab_var_mean_d{dim}",
        max_num_configs=1,
    )


def run(*inputs):
    x, = inputs
    out0 = torch.empty((x.shape[0],), device=x.device, dtype=x.dtype)
    out1 = torch.empty((x.shape[0],), device=x.device, dtype=x.dtype)
    dim = x.shape[-1]
    _kernel(dim)(x, out0, out1, dim)
    return out0, out1
