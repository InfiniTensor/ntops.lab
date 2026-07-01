import functools

import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size


def _pool3d_arrangement(input, output, kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w):
    block = block_size()
    input_arranged = input.pad(((0, 0), (0, 0), (pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w)))
    input_arranged = input_arranged.tile(
        (1, 1, kernel_d, kernel_h, kernel_w),
        strides=(-1, -1, stride_d, stride_h, stride_w),
        floor_mode=True,
    )
    input_arranged = input_arranged.ravel()
    input_arranged = input_arranged.flatten(end_dim=5).flatten(start_dim=1)
    input_arranged = input_arranged.tile((block, -1))

    output_arranged = output.tile((1, 1, 1, 1, 1))
    output_arranged = output_arranged.ravel()
    output_arranged = output_arranged.flatten(end_dim=5).flatten(start_dim=1)
    output_arranged = output_arranged.tile((block, -1))
    output_arranged.dtype = output_arranged.dtype.squeeze(1)
    return input_arranged, output_arranged


def _avg_pool3d_application(input, output):
    output = ntl.sum(input, axis=-1) / input.shape[-1]


def _max_pool3d_application(input, output):
    output = ntl.max(input, axis=-1)


@functools.cache
def _make_pool3d_kernel(kind, kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w):
    app = _avg_pool3d_application if kind == "avg" else _max_pool3d_application
    return ninetoothed.make(
        functools.partial(
            _pool3d_arrangement,
            kernel_d=kernel_d,
            kernel_h=kernel_h,
            kernel_w=kernel_w,
            stride_d=stride_d,
            stride_h=stride_h,
            stride_w=stride_w,
            pad_d=pad_d,
            pad_h=pad_h,
            pad_w=pad_w,
        ),
        app,
        (Tensor(5, other=float("-inf") if kind == "max" else None), Tensor(5)),
        kernel_name=(
            f"ntops_lab_{kind}_pool3d_"
            f"{kernel_d}x{kernel_h}x{kernel_w}_"
            f"s{stride_d}x{stride_h}x{stride_w}_"
            f"p{pad_d}x{pad_h}x{pad_w}"
        ),
    )


def _pool3d(x, kind, kernel_size=(2, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)):
    kd, kh, kw = kernel_size
    sd, sh, sw = stride
    pd, ph, pw = padding
    n, c, d, h, w = x.shape
    out_d = (d + 2 * pd - kd) // sd + 1
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1
    out = torch.empty((n, c, out_d, out_h, out_w), device=x.device, dtype=x.dtype)
    _make_pool3d_kernel(kind, kd, kh, kw, sd, sh, sw, pd, ph, pw)(x, out)
    return out


def run(*inputs):
    (x,) = inputs
    return _pool3d(x, "avg", kernel_size=(2, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))


def _compare(actual, expected, atol, rtol):
    if isinstance(actual, tuple):
        value_ok = torch.allclose(actual[0], expected[0], atol=atol, rtol=rtol)
        index_ok = torch.equal(actual[1], expected[1])
        return value_ok and index_ok, (actual[0] - expected[0]).abs().max().item()
    return torch.allclose(actual, expected, atol=atol, rtol=rtol), (actual - expected).abs().max().item()
