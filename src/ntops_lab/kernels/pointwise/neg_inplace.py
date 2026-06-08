import torch
import ninetoothed

from ninetoothed import Tensor, block_size

BLOCK_SIZE = block_size()
def arrangement(x, out):
    return x.tile((BLOCK_SIZE,)), out.tile((BLOCK_SIZE,))

def application(x, out):
    out = -x

kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1)), kernel_name="fg_split_neg_")

def run(*inputs):
    out = torch.empty_like(inputs[0])
    kernel(*inputs, out)
    return out
