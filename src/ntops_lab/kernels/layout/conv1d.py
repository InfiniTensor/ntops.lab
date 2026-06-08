import functools

import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

INPUT_PRECISION_IEEE = 2

def _mm_arrangement(input, other, output, input_precision, block_size_m=16, block_size_n=16, block_size_k=16):
    output_arranged = output.tile((block_size_m, block_size_n))

    input_arranged = input.tile((block_size_m, block_size_k))
    input_arranged = input_arranged.tile((1, -1))
    input_arranged = input_arranged.expand((-1, output_arranged.shape[1]))
    input_arranged.dtype = input_arranged.dtype.squeeze(0)

    other_arranged = other.tile((block_size_k, block_size_n))
    other_arranged = other_arranged.tile((-1, 1))
    other_arranged = other_arranged.expand((output_arranged.shape[0], -1))
    other_arranged.dtype = other_arranged.dtype.squeeze(1)

    return input_arranged, other_arranged, output_arranged, input_precision

def _mm_application(input, other, output, input_precision):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)
    if input_precision == 2:
        input_precision_: ntl.constexpr = "ieee"
    else:
        input_precision_: ntl.constexpr = "tf32"
    for k in range(input.shape[0]):
        accumulator += ntl.dot(input[k], other[k], input_precision=input_precision_)
    output = accumulator

def _conv2d_arrangement(input, weight, bias, output, input_precision, pad_h, pad_w):
    input_arranged = input.pad(((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))
    input_arranged = input_arranged.tile((1, *weight.shape[1:]), strides=(-1, -1, 1, 1), dilation=(1, 1, 1, 1), floor_mode=True)
    input_arranged = input_arranged.squeeze(1)
    input_arranged.dtype = input_arranged.dtype.squeeze(0)
    input_arranged = input_arranged.ravel()
    input_arranged = input_arranged.flatten(end_dim=3).flatten(start_dim=1)

    weight_arranged = weight.flatten(start_dim=1).permute((1, 0))

    bias_arranged = bias[None, :, None, None].expand((output.shape[0], -1, output.shape[2], output.shape[3]))
    bias_arranged = bias_arranged.permute((0, 2, 3, 1)).flatten(end_dim=3)

    output_arranged = output.permute((0, 2, 3, 1)).flatten(end_dim=3)

    bias_arranged = bias_arranged.tile((16, 16))
    input_arranged, weight_arranged, output_arranged, input_precision_arranged = _mm_arrangement(input_arranged, weight_arranged, output_arranged, input_precision)
    return input_arranged, weight_arranged, bias_arranged, output_arranged, input_precision_arranged

def _conv2d_application(input, weight, bias, output, input_precision):
    mm_output = ntl.zeros(output.shape, dtype=ntl.float32)
    _mm_application(input, weight, mm_output, input_precision)
    output = mm_output + bias

@functools.cache
def _make_conv2d_kernel(pad_h, pad_w):
    tensors = (
        Tensor(4),
        Tensor(4),
        Tensor(1),
        Tensor(4),
        Tensor(0, constexpr=True, value=INPUT_PRECISION_IEEE),
    )
    return ninetoothed.make(
        functools.partial(_conv2d_arrangement, pad_h=pad_h, pad_w=pad_w),
        _conv2d_application,
        tensors,
        kernel_name=f"ntops_lab_conv1d_conv2d_core_{pad_h}_{pad_w}",
        max_num_configs=1,
    )

def _conv2d(input, weight, bias=None, padding=(0, 0)):
    if bias is None:
        bias = torch.zeros((weight.shape[0],), device=input.device, dtype=input.dtype)
    pad_h, pad_w = padding
    n, _, h, w = input.shape
    k, _, r, s = weight.shape
    out_h = h + 2 * pad_h - r + 1
    out_w = w + 2 * pad_w - s + 1
    out = torch.empty((n, k, out_h, out_w), device=input.device, dtype=input.dtype)
    _make_conv2d_kernel(pad_h, pad_w)(input, weight, bias, out, INPUT_PRECISION_IEEE)
    return out

def run(*inputs):
    x, w, b = inputs
    y = _conv2d(x[:, :, None, :], w[:, :, None, :], b, padding=(0, 1))
    return y[:, :, 0, :]
