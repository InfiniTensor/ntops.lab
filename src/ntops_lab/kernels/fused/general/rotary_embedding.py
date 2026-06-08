import torch
import ninetoothed

from ninetoothed import Tensor


def arrangement(x0, x1, cos, sin, out0, out1):
    return x0.tile((256,)), x1.tile((256,)), cos.tile((256,)), sin.tile((256,)), out0.tile((256,)), out1.tile((256,))

def application(x0, x1, cos, sin, out0, out1):
    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos

kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1), Tensor(1), Tensor(1), Tensor(1), Tensor(1)), kernel_name="fg_fused_rotary_embedding", max_num_configs=1)

def run(*inputs):
    x0, x1, cos, sin = inputs
    out0 = torch.empty_like(x0)
    out1 = torch.empty_like(x1)
    kernel(x0, x1, cos, sin, out0, out1)
    return out0, out1
