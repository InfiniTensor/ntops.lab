import functools

import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(x, y, out, dim):
    return x.tile((1, dim.value)), y.tile((1, dim.value)), out.tile((1,)), dim


def application(x, y, out, dim):
    diff = x - y + 1.0e-6
    out = ntl.sqrt(ntl.sum(diff * diff, axis=1))


@functools.cache
def _kernel(dim):
    dim_tensor = Tensor(0, constexpr=True, value=dim, name="dim")
    return ninetoothed.make(
        arrangement,
        application,
        (Tensor(2), Tensor(2), Tensor(1), dim_tensor),
        kernel_name=f"ntops_lab_pairwise_distance_d{dim}",
        max_num_configs=1,
    )


def run(*inputs):
    x, y = inputs
    out = torch.empty((x.shape[0],), device=x.device, dtype=x.dtype)
    dim = x.shape[-1]
    _kernel(dim)(x, y, out, dim)
    return out
