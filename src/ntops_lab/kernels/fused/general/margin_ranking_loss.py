import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BLOCK = block_size()

def arrangement(x1, x2, label, out):
    return x1.tile((BLOCK,)), x2.tile((BLOCK,)), label.tile((BLOCK,)), out.tile((BLOCK,))

def application(x1, x2, label, out):
    out = ntl.maximum(0.0, 0.0 - label * (x1 - x2) + 1.0)

kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1), Tensor(1), Tensor(1)), kernel_name="ntops_lab_margin_ranking_loss")

def run(*inputs):
    x1, x2, label = inputs
    out = torch.empty_like(inputs[0])
    kernel(x1, x2, label, out)
    return out
