import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BLOCK_SIZE = block_size()
def arrangement(x, y, out):
    return x.tile((BLOCK_SIZE,)), y.tile((BLOCK_SIZE,)), out.tile((BLOCK_SIZE,))

def application(x, y, out):
    # Derive the quotient from the remainder before rounding near integers.
    remainder = ntl.libdevice.fmod(x, y)
    quotient = (x - remainder) / y
    quotient = ntl.where(
        (remainder != 0.0) & ((y < 0.0) != (remainder < 0.0)),
        quotient - 1.0,
        quotient,
    )
    rounded = ntl.floor(quotient)
    rounded = ntl.where(quotient - rounded > 0.5, rounded + 1.0, rounded)
    signed_zero = ntl.libdevice.copysign(0.0, x / y)
    out = ntl.where(y == 0.0, x / y, ntl.where(quotient == 0.0, signed_zero, rounded))

kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1), Tensor(1)), kernel_name="fg_split_floor_divide_")

def run(*inputs):
    out = torch.empty_like(inputs[0])
    kernel(*inputs, out)
    return out
