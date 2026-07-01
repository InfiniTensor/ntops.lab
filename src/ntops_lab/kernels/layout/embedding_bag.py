import functools

import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(index, weight, out, vocab, dim, bag):
    index_arr = index[:, None, :, None].expand((-1, dim.value, -1, vocab.value)).flatten(start_dim=2).flatten(end_dim=2)
    weight_arr = weight.permute((1, 0))[None, :, None, :].expand((index.shape[0], -1, bag.value, -1)).flatten(start_dim=2).flatten(end_dim=2)
    return index_arr.tile((1, bag.value * vocab.value)), weight_arr.tile((1, bag.value * vocab.value)), out.flatten().tile((1,)), vocab, dim, bag


def application(index, weight, out, vocab, dim, bag):
    out = ntl.sum(weight * (index == weight.offsets(0)), axis=1)


@functools.cache
def _kernel(vocab, dim, bag):
    vocab_tensor = Tensor(0, constexpr=True, value=vocab, name="vocab")
    dim_tensor = Tensor(0, constexpr=True, value=dim, name="dim")
    bag_tensor = Tensor(0, constexpr=True, value=bag, name="bag")
    return ninetoothed.make(
        arrangement,
        application,
        (Tensor(2), Tensor(2), Tensor(2), vocab_tensor, dim_tensor, bag_tensor),
        kernel_name=f"ntops_lab_embedding_bag_v{vocab}_d{dim}_b{bag}",
        max_num_configs=1,
    )


def run(*inputs, offsets=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, mode="sum", sparse=False, per_sample_weights=None, include_last_offset=False, padding_idx=None):
    index, weight = inputs
    if offsets is not None or max_norm is not None or scale_grad_by_freq or sparse or per_sample_weights is not None or include_last_offset or padding_idx is not None or mode != "sum":
        raise ValueError("embedding_bag currently supports 2D input, offsets=None, mode='sum', no weights/options")
    vocab, dim = weight.shape
    bag = index.shape[1]
    out = torch.empty((index.shape[0], dim), device=weight.device, dtype=weight.dtype)
    _kernel(vocab, dim, bag)(index, weight, out, vocab, dim, bag)
    return out
