import functools

import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(x, out, dim):
    return x.tile((1, 1, dim.value)), out.tile((1, 1, dim.value)), dim


def application(x, out, dim):
    mean = ntl.sum(x, axis=2) / dim
    mean_square = ntl.sum(x * x, axis=2) / dim
    var = mean_square - mean * mean
    out = (x - mean[:, :, None]) * ntl.rsqrt(var[:, :, None] + 1.0e-5)


@functools.cache
def _kernel(dim):
    dim_tensor = Tensor(0, constexpr=True, value=dim, name="dim")
    return ninetoothed.make(
        arrangement,
        application,
        (Tensor(3), Tensor(3), dim_tensor),
        kernel_name=f"ntops_lab_instance_norm_d{dim}",
        max_num_configs=1,
    )


def run(*inputs):
    x, = inputs
    out = torch.empty_like(x)
    dim = x.shape[-1]
    _kernel(dim)(x, out, dim)
    return out
