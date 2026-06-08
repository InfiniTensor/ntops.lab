import torch
from ntops_lab.kernels.linear.mm_out import run

OP_NAME = "mm_out"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(device="cuda", dtype=torch.float16):
    a = torch.randn((32, 32), device=device, dtype=dtype)
    b = torch.randn((32, 32), device=device, dtype=dtype)
    return a, b

def run_pytorch(*inputs):
    a, b = inputs
    return a @ b

def check(atol=1.0e-1, rtol=1.0e-1):
    inputs = make_inputs()
    actual = run(*inputs)
    expected = run_pytorch(*inputs)
    torch.cuda.synchronize()
    max_abs_error = (actual - expected).abs().max().item()
    passed = torch.allclose(actual, expected, atol=atol, rtol=rtol)
    print(OP_NAME, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max_abs_error)
    if not passed:
        raise AssertionError(OP_NAME)
