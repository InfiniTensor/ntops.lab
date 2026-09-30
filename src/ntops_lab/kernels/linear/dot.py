import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(x, y, out):
    return x.unsqueeze(0).tile((1, -1)), y.unsqueeze(0).tile((1, -1)), out.tile((1,))

def application(x, y, out):
    out = ntl.sum(x * y, axis=1)

kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1), Tensor(1)), kernel_name="ntops_lab_dot")

def run(*inputs):
    x, y = inputs
    out = torch.empty((1,), device=x.device, dtype=x.dtype)
    kernel(x, y, out)
    return out
