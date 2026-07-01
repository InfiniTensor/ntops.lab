import functools

import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


@functools.cache
def _kernel(channels, out_h, out_w, kh, kw, pad):
    l_cols = out_h * out_w
    reduce = channels * kh * kw * l_cols

    def arrangement(x, src_c, src_h, src_w, tgt_c, tgt_h, tgt_w, out):
        x_arr = x[:, None, None, None, :, :].expand((-1, channels, out_h, out_w, -1, -1)).flatten(start_dim=4)
        src_c_arr = src_c[None, None, None, None, :].expand((x.shape[0], channels, out_h, out_w, -1))
        src_h_arr = src_h[None, None, None, None, :].expand((x.shape[0], channels, out_h, out_w, -1))
        src_w_arr = src_w[None, None, None, None, :].expand((x.shape[0], channels, out_h, out_w, -1))
        tgt_c_arr = tgt_c.unsqueeze(4).expand((-1, -1, -1, -1, reduce))
        tgt_h_arr = tgt_h.unsqueeze(4).expand((-1, -1, -1, -1, reduce))
        tgt_w_arr = tgt_w.unsqueeze(4).expand((-1, -1, -1, -1, reduce))
        out_arr = out.unsqueeze(4).expand((-1, -1, -1, -1, reduce)).tile((1, channels, out_h, out_w, reduce))
        out_arr.dtype = out_arr.dtype.squeeze(4)
        return (
            x_arr.tile((1, channels, out_h, out_w, reduce)),
            src_c_arr.tile((1, channels, out_h, out_w, reduce)),
            src_h_arr.tile((1, channels, out_h, out_w, reduce)),
            src_w_arr.tile((1, channels, out_h, out_w, reduce)),
            tgt_c_arr.tile((1, channels, out_h, out_w, reduce)),
            tgt_h_arr.tile((1, channels, out_h, out_w, reduce)),
            tgt_w_arr.tile((1, channels, out_h, out_w, reduce)),
            out_arr,
        )

    def application(x, src_c, src_h, src_w, tgt_c, tgt_h, tgt_w, out):
        mask = (src_c == tgt_c) * (src_h == tgt_h) * (src_w == tgt_w)
        out = ntl.sum(x * mask, axis=4)

    return ninetoothed.make(
        arrangement,
        application,
        (Tensor(3), Tensor(1), Tensor(1), Tensor(1), Tensor(4), Tensor(4), Tensor(4), Tensor(4)),
        kernel_name=f"ntops_lab_fold_c{channels}_out{out_h}x{out_w}_k{kh}x{kw}_p{pad}",
        max_num_configs=1,
    )


@functools.cache
def _metadata(channels, out_h, out_w, kh, kw, pad, device):
    l_cols = out_h * out_w
    patch = torch.arange(channels * kh * kw, device=device, dtype=torch.long)
    cols = torch.arange(l_cols, device=device, dtype=torch.long)
    patch = patch[:, None].expand(channels * kh * kw, l_cols).reshape(-1)
    cols = cols[None, :].expand(channels * kh * kw, l_cols).reshape(-1)
    src_c = patch // (kh * kw)
    kk = patch % (kh * kw)
    kernel_h = kk // kw
    kernel_w = kk % kw
    col_h = cols // out_w
    col_w = cols % out_w
    src_h = col_h + kernel_h - pad
    src_w = col_w + kernel_w - pad
    return src_c, src_h, src_w


def run(*inputs, output_size=(4, 4), kernel_size=(3, 3), dilation=1, padding=1, stride=1):
    (x,) = inputs
    if dilation != 1 or stride != 1:
        raise ValueError("fold currently supports dilation=1 and stride=1")
    if isinstance(kernel_size, int):
        kh = kw = kernel_size
    else:
        kh, kw = int(kernel_size[0]), int(kernel_size[1])
    if isinstance(padding, int):
        pad = padding
    else:
        if padding[0] != padding[1]:
            raise ValueError("fold currently supports equal height/width padding")
        pad = int(padding[0])
    out_h, out_w = int(output_size[-2]), int(output_size[-1])
    channels = x.shape[1] // (kh * kw)
    out = torch.empty((x.shape[0], channels, out_h, out_w), device=x.device, dtype=x.dtype)
    src_c, src_h, src_w = _metadata(channels, out_h, out_w, kh, kw, pad, x.device)
    tgt_c = torch.arange(channels, device=x.device, dtype=torch.long).view(1, channels, 1, 1).expand(x.shape[0], channels, out_h, out_w)
    tgt_h = torch.arange(out_h, device=x.device, dtype=torch.long).view(1, 1, out_h, 1).expand(x.shape[0], channels, out_h, out_w)
    tgt_w = torch.arange(out_w, device=x.device, dtype=torch.long).view(1, 1, 1, out_w).expand(x.shape[0], channels, out_h, out_w)
    _kernel(channels, out_h, out_w, kh, kw, pad)(x, src_c, src_h, src_w, tgt_c, tgt_h, tgt_w, out)
    return out
