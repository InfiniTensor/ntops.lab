import torch
import ninetoothed
from ninetoothed import Tensor

N = 512

def arrangement(out):
    rows = out.tile((1, N))
    return rows

def application(out):
    out = out.offsets(0) == out.offsets(1)

kernel = ninetoothed.make(arrangement, application, (Tensor(2),), kernel_name="fg_extra_eye_m", max_num_configs=1)

def run(*inputs, device="cuda", dtype=torch.float32):
    out = torch.empty((N, N), device=device, dtype=dtype)
    kernel(out)
    return out
