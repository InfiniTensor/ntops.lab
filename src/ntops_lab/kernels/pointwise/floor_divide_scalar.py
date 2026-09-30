import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BLOCK_SIZE = block_size()

def arrangement(x, y, out):
    return x.tile((BLOCK_SIZE,)), y.tile((BLOCK_SIZE,)), out.tile((BLOCK_SIZE,))

def application(x, y, out):
    # Derive the quotient from the remainder before rounding near integers.
    remainder = ntl.libdevice.fmod(x, 1.25)
    quotient = (x - remainder) / 1.25
    quotient = ntl.where(
        (remainder != 0.0) & ((1.25 < 0.0) != (remainder < 0.0)),
        quotient - 1.0,
        quotient,
    )
    rounded = ntl.floor(quotient)
    rounded = ntl.where(quotient - rounded > 0.5, rounded + 1.0, rounded)
    signed_zero = ntl.libdevice.copysign(0.0, x / 1.25)
    out = ntl.where(1.25 == 0.0, x / 1.25, ntl.where(quotient == 0.0, signed_zero, rounded))

kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1), Tensor(1)), kernel_name="ntops_lab_floor_divide_scalar")

def run(*inputs):
    x, y = inputs
    out = torch.empty((x.numel(),), device=x.device, dtype=x.dtype)
    kernel(x, y, out)
    return out
