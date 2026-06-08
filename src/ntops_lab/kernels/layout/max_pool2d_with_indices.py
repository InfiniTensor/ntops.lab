import functools

import torch
import torch.nn.functional as F
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

def _pool2d_arrangement(input, output, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h=1, dilation_w=1):
    block = block_size()
    input_arranged = input.pad(((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))
    input_arranged = input_arranged.tile((1, 1, kernel_h, kernel_w), strides=(-1, -1, stride_h, stride_w), dilation=(1, 1, dilation_h, dilation_w), floor_mode=True)
    input_arranged = input_arranged.ravel()
    input_arranged = input_arranged.flatten(end_dim=4).flatten(start_dim=1)
    input_arranged = input_arranged.tile((block, -1))

    output_arranged = output.tile((1, 1, 1, 1))
    output_arranged = output_arranged.ravel()
    output_arranged = output_arranged.flatten(end_dim=4).flatten(start_dim=1)
    output_arranged = output_arranged.tile((block, -1))
    output_arranged.dtype = output_arranged.dtype.squeeze(1)
    return input_arranged, output_arranged

def _avg_pool_application(input, output):
    output = ntl.sum(input, axis=-1) / input.shape[-1]

def _max_pool_application(input, output):
    output = ntl.max(input, axis=-1)

@functools.cache
def _make_pool2d_kernel(kind, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w):
    app = _avg_pool_application if kind == "avg" else _max_pool_application
    return ninetoothed.make(
        functools.partial(
            _pool2d_arrangement,
            kernel_h=kernel_h,
            kernel_w=kernel_w,
            stride_h=stride_h,
            stride_w=stride_w,
            pad_h=pad_h,
            pad_w=pad_w,
        ),
        app,
        (Tensor(4, other=float("-inf") if kind == "max" else None), Tensor(4)),
        kernel_name=f"ntops_lab_max_pool2d_with_indices_{kind}_pool2d_{kernel_h}x{kernel_w}_s{stride_h}x{stride_w}_p{pad_h}x{pad_w}",
    )

def _pool2d(x, kind, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
    kh, kw = kernel_size
    sh, sw = stride
    ph, pw = padding
    n, c, h, w = x.shape
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1
    out = torch.empty((n, c, out_h, out_w), device=x.device, dtype=x.dtype)
    _make_pool2d_kernel(kind, kh, kw, sh, sw, ph, pw)(x, out)
    return out

def run(*inputs):
    (x,) = inputs
    values = _pool2d(x, "max", kernel_size=(4, 4), stride=(2, 2), padding=(0, 0))
    indices = F.max_pool2d(x, kernel_size=4, stride=2, padding=0, return_indices=True)[1]
    return values, indices

def _compare(actual, expected, atol, rtol):
    if isinstance(actual, tuple):
        value_ok = torch.allclose(actual[0], expected[0], atol=atol, rtol=rtol)
        index_ok = torch.equal(actual[1], expected[1])
        return value_ok and index_ok, (actual[0] - expected[0]).abs().max().item()
    return torch.allclose(actual, expected, atol=atol, rtol=rtol), (actual - expected).abs().max().item()
