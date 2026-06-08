import torch

from ntops_lab.kernels.linear.outer import run

OP_NAME = "outer"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(m=32, n=32, device="cuda", dtype=torch.float16):
    x = torch.randn((m,), device=device, dtype=dtype)
    y = torch.randn((n,), device=device, dtype=dtype)
    return x, y

def run_pytorch(*inputs):
    x, y = inputs
    return torch.outer(x, y)

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
