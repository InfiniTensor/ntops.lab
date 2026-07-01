import functools

import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(x1, x2, label, out, dim):
    return x1.tile((1, dim.value)), x2.tile((1, dim.value)), label.tile((1,)), out.tile((1,)), dim


def application(x1, x2, label, out, dim):
    xx = ntl.sum(x1 * x1, axis=1)
    yy = ntl.sum(x2 * x2, axis=1)
    xy = ntl.sum(x1 * x2, axis=1)
    cos = xy / ntl.maximum(ntl.sqrt(xx) * ntl.sqrt(yy), 1.0e-8)
    out = ntl.where(label > 0.0, 1.0 - cos, ntl.maximum(0.0, cos))


@functools.cache
def _kernel(dim):
    dim_tensor = Tensor(0, constexpr=True, value=dim, name="dim")
    return ninetoothed.make(
        arrangement,
        application,
        (Tensor(2), Tensor(2), Tensor(1), Tensor(1), dim_tensor),
        kernel_name=f"ntops_lab_cosine_embedding_loss_d{dim}",
        max_num_configs=1,
    )


def run(*inputs):
    x1, x2, label = inputs
    out = torch.empty((x1.shape[0],), device=x1.device, dtype=x1.dtype)
    dim = x1.shape[-1]
    _kernel(dim)(x1, x2, label, out, dim)
    return out
