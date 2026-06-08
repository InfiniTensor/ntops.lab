import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BLOCK_SIZE = block_size()
def arrangement(a, b, out):
    return a.tile((BLOCK_SIZE,)), b.tile((BLOCK_SIZE,)), out.tile((BLOCK_SIZE,))

def application(a, b, out):
    out = 0.5 * a * (1.0 + (2.0 / (1.0 + ntl.exp(-2.0 * (0.7978845608028654 * (a + 0.044715 * a * a * a)))) - 1.0)) * b

kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1), Tensor(1)), kernel_name="fg_fused_geglu")

def run(*inputs):
    a, b = inputs
    out = torch.empty_like(a)
    kernel(a, b, out)
    return out
