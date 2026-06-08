import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BLOCK_SIZE = block_size()
def arrangement(x, y, z, out):
    return x.tile((BLOCK_SIZE,)), y.tile((BLOCK_SIZE,)), z.tile((BLOCK_SIZE,)), out.tile((BLOCK_SIZE,))

def application(x, y, z, out):
    out = ntl.where(x != 0.0, y, z)

kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1), Tensor(1), Tensor(1)), kernel_name="fg_split_where_scalar_other")

def run(*inputs):
    out = torch.empty_like(inputs[0], dtype=inputs[1].dtype)
    kernel(*inputs, out)
    return out
