import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BLOCK_SIZE = block_size()

def arrangement(x, y, z, out):
    return x.tile((BLOCK_SIZE,)), y.tile((BLOCK_SIZE,)), z.tile((BLOCK_SIZE,)), out.tile((BLOCK_SIZE,))

def application(x, y, z, out):
    out = ntl.where(x - y < 0.0, y - x, x - y) <= (1.0e-5 + 1.0e-5 * ntl.where(y < 0.0, -y, y))

kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1), Tensor(1), Tensor(1)), kernel_name="ntops_lab_isclose")

def run(*inputs):
    x, y, z = inputs
    out = torch.empty((x.numel(),), device=x.device, dtype=torch.bool)
    kernel(x, y, z, out)
    return out
