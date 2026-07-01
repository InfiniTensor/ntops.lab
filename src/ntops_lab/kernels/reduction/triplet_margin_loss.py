import functools

import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(anchor, positive, negative, out, dim):
    return anchor.tile((1, dim.value)), positive.tile((1, dim.value)), negative.tile((1, dim.value)), out.tile((1,)), dim


def application(anchor, positive, negative, out, dim):
    ap = anchor - positive + 1.0e-6
    an = anchor - negative + 1.0e-6
    d_ap = ntl.sqrt(ntl.sum(ap * ap, axis=1))
    d_an = ntl.sqrt(ntl.sum(an * an, axis=1))
    out = ntl.maximum(0.0, d_ap - d_an + 1.0)


@functools.cache
def _kernel(dim):
    dim_tensor = Tensor(0, constexpr=True, value=dim, name="dim")
    return ninetoothed.make(
        arrangement,
        application,
        (Tensor(2), Tensor(2), Tensor(2), Tensor(1), dim_tensor),
        kernel_name=f"ntops_lab_triplet_margin_loss_d{dim}",
        max_num_configs=1,
    )


def run(*inputs):
    anchor, positive, negative = inputs
    out = torch.empty((anchor.shape[0],), device=anchor.device, dtype=anchor.dtype)
    dim = anchor.shape[-1]
    _kernel(dim)(anchor, positive, negative, out, dim)
    return out
