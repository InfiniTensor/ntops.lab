import torch
from ntops_lab.kernels.pointwise.atan2 import run

OP_NAME = "atan2"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"
DEFAULT_NUMEL = 1048576

def make_inputs(numel=DEFAULT_NUMEL, device="cuda", dtype=torch.float32):
    y = torch.randn((numel,), device=device, dtype=dtype)
    x = torch.randn((numel,), device=device, dtype=dtype)
    return y, x

def run_pytorch(*inputs):
    y, x = inputs
    return torch.atan2(y, x)

def check(atol=2.0e-3, rtol=2.0e-3):
    inputs = make_inputs()
    actual = run(*inputs)
    expected = run_pytorch(*inputs)
    torch.cuda.synchronize()
    max_abs_error = (actual - expected).abs().max().item()
    passed = torch.allclose(actual, expected, atol=atol, rtol=rtol)
    print(OP_NAME, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max_abs_error)
    if not passed:
        raise AssertionError(OP_NAME)
