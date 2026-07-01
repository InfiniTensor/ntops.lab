import torch
import torch.nn.functional as F
from ntops_lab.kernels.layout.max_unpool3d import run

OP_NAME = "max_unpool3d"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"
SIZES = ((8, 8, 8), (6, 8, 10))


def make_inputs(size, device="cuda", dtype=torch.float32):
    base = torch.randn((2, 2, size[0], size[1], size[2]), device=device, dtype=dtype) * 0.5
    values, indices = F.max_pool3d(base, kernel_size=2, stride=2, return_indices=True)
    return values, indices


def run_pytorch(*inputs, output_size):
    values, indices = inputs
    return F.max_unpool3d(values, indices, kernel_size=2, stride=2, padding=0, output_size=(values.shape[0], values.shape[1], *output_size))


def check(atol=1.0e-6, rtol=1.0e-6):
    for size in SIZES:
        inputs = make_inputs(size)
        actual = run(*inputs, output_size=(inputs[0].shape[0], inputs[0].shape[1], *size))
        expected = run_pytorch(*inputs, output_size=size)
        torch.cuda.synchronize()
        max_abs_error = (actual - expected).abs().max().item()
        passed = torch.allclose(actual, expected, atol=atol, rtol=rtol)
        print(OP_NAME, "size=", size, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max_abs_error)
        if not passed:
            raise AssertionError(OP_NAME)
