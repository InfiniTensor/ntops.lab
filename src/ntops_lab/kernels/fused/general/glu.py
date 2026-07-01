import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BLOCK = block_size()

def arrangement(a, b, out):
    return a.flatten().tile((BLOCK,)), b.flatten().tile((BLOCK,)), out.flatten().tile((BLOCK,))

def application(a, b, out):
    out = a * ntl.sigmoid(b)

kernel = ninetoothed.make(arrangement, application, (Tensor(2), Tensor(2), Tensor(2)), kernel_name="ntops_lab_glu")

def run(*inputs):
    x, = inputs
    half = x.shape[1] // 2
    a = x[:, :half]
    b = x[:, half:]
    out = torch.empty_like(a)
    kernel(a, b, out)
    return out
