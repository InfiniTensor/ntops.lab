import functools

import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


@functools.cache
def _kernel(classes):
    def arrangement(x, target_ids, class_ids, out):
        return x.tile((1, classes)), target_ids.tile((1, classes)), class_ids.tile((1, classes)), out.tile((1,))

    def application(x, target_ids, class_ids, out):
        selected = ntl.sum(x * (target_ids == class_ids), axis=1)
        out = 0.0 - selected

    return ninetoothed.make(
        arrangement,
        application,
        (Tensor(2), Tensor(2), Tensor(2), Tensor(1)),
        kernel_name=f"ntops_lab_nll_loss_c{classes}",
        max_num_configs=1,
    )


def run(*inputs):
    x, target = inputs
    classes = x.shape[1]
    target_ids = target.view(-1, 1).repeat(1, classes)
    class_ids = torch.arange(classes, device=target.device, dtype=target.dtype).view(1, classes).repeat(x.shape[0], 1)
    out = torch.empty((x.shape[0],), device=x.device, dtype=x.dtype)
    _kernel(classes)(x, target_ids, class_ids, out)
    return out
