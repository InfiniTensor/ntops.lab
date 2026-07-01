import functools

import torch
import ninetoothed
from ninetoothed import Tensor


@functools.cache
def _kernel(rows, cols):
    def arrangement(out):
        return out.tile((1, cols))

    def application(out):
        out = out.offsets(0) == out.offsets(1)

    return ninetoothed.make(
        arrangement,
        application,
        (Tensor(2),),
        kernel_name=f"fg_extra_eye_m_{rows}x{cols}",
        max_num_configs=1,
    )


def run(*inputs, n=512, m=None, device="cuda", dtype=torch.float32):
    rows = int(n)
    cols = rows if m is None else int(m)
    out = torch.empty((rows, cols), device=device, dtype=dtype)
    _kernel(rows, cols)(out)
    return out
