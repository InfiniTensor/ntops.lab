import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

HIDDEN = 32

def arrangement(x, residual, weight, out):
    weight = weight[None, :].expand((x.shape[0], -1))
    return x.tile((1, HIDDEN)), residual.tile((1, HIDDEN)), weight.tile((1, HIDDEN)), out.tile((1, HIDDEN))

def application(x, residual, weight, out):
    y = x + residual
    rrms = ntl.rsqrt(ntl.sum(y * y, axis=1) / 32.0 + 1.0e-5)
    out = y * rrms[:, None] * weight

kernel = ninetoothed.make(arrangement, application, (Tensor(2), Tensor(2), Tensor(1), Tensor(2)), kernel_name="fg_fused_add_rms_norm", max_num_configs=1)

def run(*inputs):
    x, residual, weight = inputs
    out = torch.empty_like(x)
    kernel(x, residual, weight, out)
    return out
