import torch
import torch.nn.functional as F
from ntops_lab.kernels.layout.conv_tbc import run

OP_NAME = "conv_tbc"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(device="cuda", dtype=torch.float16):
    x = torch.randn((16, 2, 2), device=device, dtype=dtype)
    w = torch.randn((3, 2, 3), device=device, dtype=dtype)
    b = torch.randn((3,), device=device, dtype=dtype)
    return x, w, b

def run_pytorch(*inputs):
    x, w, b = inputs
    return F.conv_tbc(x, w, b, pad=1)

def check(atol=1.0e-1, rtol=1.0e-1):
    inputs = make_inputs()
    actual = run(*inputs)
    expected = run_pytorch(*inputs)
    torch.cuda.synchronize()
    if actual.dtype == torch.bool or not torch.is_floating_point(expected):
        passed = torch.equal(actual, expected)
        max_abs_error = 0.0 if passed else (actual.to(torch.float32) - expected.to(torch.float32)).abs().max().item()
    else:
        max_abs_error = (actual - expected).abs().max().item()
        passed = torch.allclose(actual, expected, atol=atol, rtol=rtol)
    print(OP_NAME, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max_abs_error)
    if not passed:
        raise AssertionError(OP_NAME)
