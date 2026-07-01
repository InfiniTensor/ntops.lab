import torch
import torch.nn.functional as F
from ntops_lab.kernels.linear.bilinear import run

OP_NAME = "bilinear"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"
SHAPES = ((8192, 8, 8, 4), (4096, 5, 7, 3))


def make_inputs(shape, device="cuda", dtype=torch.float32):
    batch, in1, in2, out_features = shape
    x1 = torch.randn((batch, in1), device=device, dtype=dtype) * 0.5
    x2 = torch.randn((batch, in2), device=device, dtype=dtype) * 0.5
    weight = torch.randn((out_features, in1, in2), device=device, dtype=dtype) * 0.5
    bias = torch.randn((out_features,), device=device, dtype=dtype) * 0.5
    return x1, x2, weight, bias


def run_pytorch(*inputs):
    x1, x2, weight, bias = inputs
    return F.bilinear(x1, x2, weight, bias)


def check(atol=1.0e-3, rtol=1.0e-3):
    for shape in SHAPES:
        inputs = make_inputs(shape)
        actual = run(*inputs)
        expected = run_pytorch(*inputs)
        torch.cuda.synchronize()
        max_abs_error = (actual - expected).abs().max().item()
        passed = torch.allclose(actual, expected, atol=atol, rtol=rtol)
        print(OP_NAME, "shape=", shape, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max_abs_error)
        if not passed:
            raise AssertionError(OP_NAME)
