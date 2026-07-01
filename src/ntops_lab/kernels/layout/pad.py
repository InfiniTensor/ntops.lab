import torch
import ninetoothed
from ninetoothed import Tensor, block_size

BLOCK = block_size()

def arrangement(x, out):
    return x.flatten().tile((BLOCK,)), out[1:-4, 2:-3].flatten().tile((BLOCK,))

def application(x, out):
    out = x

kernel = ninetoothed.make(arrangement, application, (Tensor(2), Tensor(2)), kernel_name="ntops_lab_pad")

def run(*inputs):
    x, = inputs
    out = torch.full((x.shape[0] + 5, x.shape[1] + 5), 0.0, device=x.device, dtype=x.dtype)
    kernel(x, out)
    return out
