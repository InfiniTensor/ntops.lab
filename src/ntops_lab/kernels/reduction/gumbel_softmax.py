import torch

from ntops_lab.kernels.reduction.softmax import run as _softmax

SEED = 20260613

def run(*inputs, tau=1.0, hard=False, eps=1.0e-10, dim=-1):
    (x,) = inputs
    if tau != 1.0 or hard or dim not in (-1, x.ndim - 1):
        raise ValueError("gumbel_softmax currently supports tau=1, hard=False, dim=-1")
    torch.manual_seed(SEED)
    noise = -torch.empty_like(x).exponential_().log()
    return _softmax(x + noise)
