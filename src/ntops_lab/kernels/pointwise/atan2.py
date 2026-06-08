import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BLOCK_SIZE = block_size()

def arrangement(y, x, out):
    return y.tile((BLOCK_SIZE,)), x.tile((BLOCK_SIZE,)), out.tile((BLOCK_SIZE,))

def _atan2_approx(y, x):
    yy = y + 0.0
    xx = x + 0.0
    pi = 3.1415927410125732
    half_pi = 1.5707963705062866
    abs_y = ntl.where(yy < 0.0, -yy, yy)
    abs_x = ntl.where(xx < 0.0, -xx, xx)
    swap_xy = abs_y > abs_x
    num = ntl.where(swap_xy, abs_x, abs_y)
    den = ntl.where(swap_xy, abs_y, abs_x)
    den_safe = ntl.where(den == 0.0, 1.0, den)
    z = num / den_safe
    z2 = z * z
    poly = z * (0.9998660 + z2 * (-0.3302995 + z2 * (0.1801410 + z2 * (-0.0851330 + z2 * 0.0208351))))
    theta = ntl.where(swap_xy, half_pi - poly, poly)
    res = ntl.where(xx < 0.0, pi - theta, theta)
    res = ntl.where(yy < 0.0, -res, res)
    return ntl.where((xx == 0.0) & (yy == 0.0), 0.0, res)

def application(y, x, out):
    out = _atan2_approx(y, x)

kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1), Tensor(1)), kernel_name="ntops_lab_atan2")

def run(*inputs):
    y, x = inputs
    out = torch.empty_like(y)
    kernel(y, x, out)
    return out
