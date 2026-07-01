import functools

import torch
import ninetoothed
from ninetoothed import Tensor


@functools.cache
def _kernel(num_classes):
    def arrangement(index, out):
        index_arr = index.unsqueeze(1).expand((-1, num_classes))
        return index_arr.tile((1, num_classes)), out.tile((1, num_classes))

    def application(index, out):
        out = index == out.offsets(1)

    return ninetoothed.make(
        arrangement,
        application,
        (Tensor(1), Tensor(2)),
        kernel_name=f"ntops_lab_one_hot_c{num_classes}",
        max_num_configs=1,
    )


def run(*inputs, num_classes=-1):
    (index,) = inputs
    if num_classes < 0:
        num_classes = int(index.max().item()) + 1
    out = torch.empty((*index.shape, num_classes), device=index.device, dtype=torch.long)
    _kernel(num_classes)(index, out)
    return out
