import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BLOCK_SIZE = block_size()
def arrangement(x, y, out):
    return x.tile((BLOCK_SIZE,)), y.tile((BLOCK_SIZE,)), out.tile((BLOCK_SIZE,))

def application(x, y, out):
    out = ntl.floor(x / y)

kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1), Tensor(1)), kernel_name="fg_split_div_mode")

def run(*inputs):
    out = torch.empty_like(inputs[0])
    kernel(*inputs, out)
    return out
