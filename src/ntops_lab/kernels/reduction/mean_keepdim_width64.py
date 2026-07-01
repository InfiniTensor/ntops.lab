import functools

import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(x, out, dim):
    return x.tile((1, dim.value)), out.tile((1, 1)), dim


def application(x, out, dim):
    out = (ntl.sum(x, axis=1) / dim)[:, None]


@functools.cache
def _kernel(dim):
    dim_tensor = Tensor(0, constexpr=True, value=dim, name="dim")
    return ninetoothed.make(
        arrangement,
        application,
        (Tensor(2), Tensor(2), dim_tensor),
        kernel_name=f"ntops_lab_mean_keepdim_width64_d{dim}",
        max_num_configs=1,
    )


def run(*inputs):
    x, = inputs
    out = torch.empty((x.shape[0], 1), device=x.device, dtype=x.dtype)
    dim = x.shape[-1]
    _kernel(dim)(x, out, dim)
    return out
