import torch

from ntops_lab.kernels.layout.conv_transpose2d import _conv2d

def run(*inputs):
    x, w, b = inputs
    n, _, d, h, width = x.shape
    _, cout, kd, kh, kw = w.shape
    out = torch.empty((n, cout, d + kd - 1, h + kh - 1, width + kw - 1), device=x.device, dtype=x.dtype)
    out.zero_()
    for id_ in range(d):
        for kz in range(kd):
            wf = w[:, :, kz, :, :].permute(1, 0, 2, 3).flip(-1).flip(-2)
            part = _conv2d(x[:, :, id_, :, :], wf, None, padding=(kh - 1, kw - 1))
            out[:, :, id_ + kz, :, :] += part
    out += b[None, :, None, None, None]
    return out
