import torch
from ntops_lab.kernels.linear.addmv import run

OP_NAME = "addmv"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(m=32, k=32, device="cuda", dtype=torch.float32):
    bias = torch.randn((m,), device=device, dtype=dtype)
    a = torch.randn((m, k), device=device, dtype=dtype)
    x = torch.randn((k,), device=device, dtype=dtype)
    return bias, a, x

def run_pytorch(*inputs):
    bias, a, x = inputs
    return bias + a @ x

def check(atol=1.0e-3, rtol=1.0e-3):
    inputs = make_inputs()
    actual = run(*inputs)
    expected = run_pytorch(*inputs)
    torch.cuda.synchronize()
    max_abs_error = (actual - expected).abs().max().item()
    passed = torch.allclose(actual, expected, atol=atol, rtol=rtol)
    print(OP_NAME, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max_abs_error)
    if not passed:
        raise AssertionError(OP_NAME)
