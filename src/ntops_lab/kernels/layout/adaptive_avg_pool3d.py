import torch
from ntops_lab.kernels.layout.avg_pool2d import _pool2d

def run(*inputs):
    (x,) = inputs
    values = []
    for d in range(0, x.shape[2], 2):
        cur = None
        for kd in range(2):
            part = _pool2d(x[:, :, d + kd, :, :], "avg", kernel_size=(4, 4), stride=(4, 4), padding=(0, 0))
            cur = part if cur is None else cur + part
        values.append((cur / 2.0)[:, :, None, :, :])
    return torch.cat(values, dim=2)
