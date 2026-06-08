import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

DIM = 32

def arrangement(x, out):
    return x.tile((1, DIM)), out.tile((1,))

def application(x, out):
    out = ntl.sum(x, axis=1) / 32.0

kernel = ninetoothed.make(arrangement, application, (Tensor(2), Tensor(1)), kernel_name="fg_extra_mean_dim", max_num_configs=1)

def run(*inputs):
    (x,) = inputs
    out = torch.empty((x.shape[0],), device=x.device, dtype=x.dtype)
    kernel(x, out)
    return out
