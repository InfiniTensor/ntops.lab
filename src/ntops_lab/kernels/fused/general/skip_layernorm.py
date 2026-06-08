import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

HIDDEN = 32

def arrangement(x, skip, gamma, beta, out):
    gamma = gamma[None, :].expand((x.shape[0], -1))
    beta = beta[None, :].expand((x.shape[0], -1))
    return x.tile((1, HIDDEN)), skip.tile((1, HIDDEN)), gamma.tile((1, HIDDEN)), beta.tile((1, HIDDEN)), out.tile((1, HIDDEN))

def application(x, skip, gamma, beta, out):
    y = x + skip
    mean = ntl.sum(y, axis=1) / 32.0
    centered = y - mean[:, None]
    var = ntl.sum(centered * centered, axis=1) / 32.0
    out = centered * ntl.rsqrt(var[:, None] + 1.0e-5) * gamma + beta

kernel = ninetoothed.make(arrangement, application, (Tensor(2), Tensor(2), Tensor(1), Tensor(1), Tensor(2)), kernel_name="fg_fused_skip_layernorm", max_num_configs=1)

def run(*inputs):
    x, skip, gamma, beta = inputs
    out = torch.empty_like(x)
    kernel(x, skip, gamma, beta, out)
    return out
