import torch
import ninetoothed

from ninetoothed import Tensor, block_size


BLOCK_SIZE = block_size()


def unary_arrangement(x, out):
    return x.tile((BLOCK_SIZE,)), out.tile((BLOCK_SIZE,))


def binary_arrangement(x, y, out):
    return x.tile((BLOCK_SIZE,)), y.tile((BLOCK_SIZE,)), out.tile((BLOCK_SIZE,))


def make_unary_kernel(application, op_name):
    return ninetoothed.make(
        unary_arrangement,
        application,
        (Tensor(1), Tensor(1)),
        kernel_name=f"ntops_lab_{op_name}",
    )


def make_binary_kernel(application, op_name):
    return ninetoothed.make(
        binary_arrangement,
        application,
        (Tensor(1), Tensor(1), Tensor(1)),
        kernel_name=f"ntops_lab_{op_name}",
    )


def run_unary(kernel, x):
    out = torch.empty((x.numel(),), device=x.device, dtype=x.dtype)
    kernel(x, out)
    return out


def run_binary(kernel, x, y):
    out = torch.empty((x.numel(),), device=x.device, dtype=x.dtype)
    kernel(x, y, out)
    return out
