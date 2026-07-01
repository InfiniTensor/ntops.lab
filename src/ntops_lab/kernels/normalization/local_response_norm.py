import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

SIZE = 5
ALPHA = 1.0e-4
BETA = 0.75
K = 1.0
BLOCK = block_size()

def sumsq_arrangement(x, sumsq):
    window = x.pad(((0, 0), (SIZE // 2, SIZE // 2), (0, 0), (0, 0)))
    window = window.tile((1, SIZE, 1, 1), strides=(-1, 1, -1, -1), floor_mode=True)
    window = window.ravel()
    window = window.flatten(end_dim=4).flatten(start_dim=1)
    window = window.tile((BLOCK, -1))
    sumsq_arr = sumsq.flatten().unsqueeze(1).tile((BLOCK, -1))
    sumsq_arr.dtype = sumsq_arr.dtype.squeeze(1)
    return window, sumsq_arr

def sumsq_application(window, sumsq):
    sumsq = ntl.sum(window * window, axis=1)

sumsq_kernel = ninetoothed.make(
    sumsq_arrangement,
    sumsq_application,
    (Tensor(4, other=0.0), Tensor(4)),
    kernel_name="ntops_lab_local_response_norm_sumsq",
)

def norm_arrangement(x, sumsq, out):
    return x.flatten().tile((BLOCK,)), sumsq.flatten().tile((BLOCK,)), out.flatten().tile((BLOCK,))

def norm_application(x, sumsq, out):
    scale = 1.0 + (1.0e-4 / 5.0) * sumsq
    out = x * ntl.exp((-0.75) * ntl.log(scale))

norm_kernel = ninetoothed.make(
    norm_arrangement,
    norm_application,
    (Tensor(4), Tensor(4), Tensor(4)),
    kernel_name="ntops_lab_local_response_norm_apply",
)

def run(*inputs, size=SIZE, alpha=ALPHA, beta=BETA, k=K):
    (x,) = inputs
    if size != SIZE or alpha != ALPHA or beta != BETA or k != K:
        raise ValueError("local_response_norm currently supports size=5, alpha=1e-4, beta=0.75, k=1.0")
    sumsq = torch.empty_like(x)
    out = torch.empty_like(x)
    sumsq_kernel(x, sumsq)
    norm_kernel(x, sumsq, out)
    return out
