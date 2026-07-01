import functools

import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


@functools.cache
def _kernel(classes):
    def arrangement(x, target_ids, class_ids, out, classes_tensor):
        return x.tile((1, classes)), target_ids.tile((1, classes)), class_ids.tile((1, classes)), out.tile((1,)), classes_tensor

    def application(x, target_ids, class_ids, out, classes_tensor):
        target_score = ntl.sum(x * (target_ids == class_ids), axis=1)
        raw = 1.0 - target_score[:, None] + x
        losses = ntl.maximum(0.0, raw) * (target_ids != class_ids)
        out = ntl.sum(losses, axis=1) / classes_tensor

    return ninetoothed.make(
        arrangement,
        application,
        (Tensor(2), Tensor(2), Tensor(2), Tensor(1), Tensor(0, constexpr=True, value=classes, name="classes")),
        kernel_name=f"ntops_lab_multi_margin_loss_c{classes}",
        max_num_configs=1,
    )


def run(*inputs, p=1, margin=1.0, weight=None, reduction="none"):
    x, target = inputs
    if p != 1 or margin != 1.0 or weight is not None or reduction != "none":
        raise ValueError("multi_margin_loss currently supports p=1, margin=1.0, weight=None, reduction='none'")
    classes = x.shape[1]
    target_ids = target.view(-1, 1).repeat(1, classes)
    class_ids = torch.arange(classes, device=target.device, dtype=target.dtype).view(1, classes).repeat(x.shape[0], 1)
    out = torch.empty((x.shape[0],), device=x.device, dtype=x.dtype)
    _kernel(classes)(x, target_ids, class_ids, out, classes)
    return out
