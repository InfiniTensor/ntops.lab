import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

BT = 64

def arrangement(x, out):
    x = x.tile((1, 1, BT), strides=(-1, -1, BT), floor_mode=True)
    out = out.tile((1, 1, BT), strides=(-1, -1, BT), floor_mode=True)
    x = x.ravel().flatten(end_dim=3).flatten(start_dim=1).tile((1, -1))
    out = out.ravel().flatten(end_dim=3).flatten(start_dim=1).tile((1, -1))
    x.dtype = x.dtype.squeeze(0)
    out.dtype = out.dtype.squeeze(0)
    return x, out

def application(x, out):
    acc = x[0] * 0.0
    for i in range(64):
        valid = x[i].offsets(-1) < x.source.shape[-1]
        acc = acc + x[i]
        out[i] = ntl.where(valid, acc, out[i])

kernel = ninetoothed.make(arrangement, application, (Tensor(3, other=0.0), Tensor(3)), kernel_name="fg_fused_fla_cumsum_bt64", max_num_configs=1)

def run(*inputs):
    (x,) = inputs
    out = torch.empty_like(x)
    kernel(x, out)
    return out
