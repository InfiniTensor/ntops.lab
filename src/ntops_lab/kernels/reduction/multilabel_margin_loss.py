import functools

import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


@functools.cache
def _score_kernel(classes):
    def score_arrangement(x, target, class_ids, score):
        x_arr = x[:, None, :].expand((-1, classes, -1)).flatten(end_dim=2)
        target_arr = target[:, :, None].expand((-1, -1, classes)).flatten(end_dim=2)
        class_arr = class_ids[:, None, :].expand((-1, classes, -1)).flatten(end_dim=2)
        return x_arr.tile((1, classes)), target_arr.tile((1, classes)), class_arr.tile((1, classes)), score.flatten().tile((1,))

    def score_application(x, target, class_ids, score):
        score = ntl.sum(x * (target == class_ids), axis=1)

    return ninetoothed.make(
        score_arrangement,
        score_application,
        (Tensor(2), Tensor(2, other=-1), Tensor(2), Tensor(2)),
        kernel_name=f"ntops_lab_multilabel_margin_loss_score_c{classes}",
        max_num_configs=1,
    )


@functools.cache
def _membership_kernel(classes):
    def membership_arrangement(target, class_ids, membership):
        target_arr = target[:, None, :].expand((-1, classes, -1)).flatten(end_dim=2)
        class_arr = class_ids[:, :, None].expand((-1, -1, classes)).flatten(end_dim=2)
        return target_arr.tile((1, classes)), class_arr.tile((1, classes)), membership.flatten().tile((1,))

    def membership_application(target, class_ids, membership):
        membership = ntl.sum(target == class_ids, axis=1) > 0

    return ninetoothed.make(
        membership_arrangement,
        membership_application,
        (Tensor(2, other=-1), Tensor(2), Tensor(2)),
        kernel_name=f"ntops_lab_multilabel_margin_loss_membership_c{classes}",
        max_num_configs=1,
    )


@functools.cache
def _partial_kernel(classes):
    def partial_arrangement(x, target, score, membership, partial):
        x_arr = x[:, None, :].expand((-1, classes, -1)).flatten(end_dim=2)
        target_arr = target[:, :, None].expand((-1, -1, classes)).flatten(end_dim=2)
        score_arr = score[:, :, None].expand((-1, -1, classes)).flatten(end_dim=2)
        membership_arr = membership[:, None, :].expand((-1, classes, -1)).flatten(end_dim=2)
        return (
            x_arr.tile((1, classes)),
            target_arr.tile((1, classes)),
            score_arr.tile((1, classes)),
            membership_arr.tile((1, classes)),
            partial.flatten().tile((1,)),
        )

    def partial_application(x, target, score, membership, partial):
        raw = 1.0 - score + x
        losses = ntl.maximum(0.0, raw) * (target >= 0) * (membership == 0)
        partial = ntl.sum(losses, axis=1)

    return ninetoothed.make(
        partial_arrangement,
        partial_application,
        (Tensor(2, other=-1.0e20), Tensor(2, other=-1), Tensor(2), Tensor(2), Tensor(2)),
        kernel_name=f"ntops_lab_multilabel_margin_loss_partial_c{classes}",
        max_num_configs=1,
    )


@functools.cache
def _reduce_kernel(classes):
    def reduce_arrangement(partial, out, classes_tensor):
        return partial.tile((1, classes)), out.tile((1,)), classes_tensor

    def reduce_application(partial, out, classes_tensor):
        out = ntl.sum(partial, axis=1) / classes_tensor

    return ninetoothed.make(
        reduce_arrangement,
        reduce_application,
        (Tensor(2), Tensor(1), Tensor(0, constexpr=True, value=classes, name="classes")),
        kernel_name=f"ntops_lab_multilabel_margin_loss_reduce_c{classes}",
        max_num_configs=1,
    )


def run(*inputs, reduction="none"):
    x, target = inputs
    if reduction != "none":
        raise ValueError("multilabel_margin_loss currently supports reduction='none'")
    classes = x.shape[1]
    class_ids = torch.arange(classes, device=target.device, dtype=target.dtype).view(1, classes).repeat(x.shape[0], 1)
    score = torch.empty_like(x)
    membership = torch.empty_like(x, dtype=torch.bool)
    partial = torch.empty_like(x)
    out = torch.empty((x.shape[0],), device=x.device, dtype=x.dtype)
    _score_kernel(classes)(x, target, class_ids, score)
    _membership_kernel(classes)(target, class_ids, membership)
    _partial_kernel(classes)(x, target, score, membership, partial)
    _reduce_kernel(classes)(partial, out, classes)
    return out
