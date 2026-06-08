import torch
from ntops_lab.kernels.linear.dot import run

OP_NAME = "dot"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(n=32, device="cuda", dtype=torch.float32):
    x = torch.randn((n,), device=device, dtype=dtype)
    y = torch.randn((n,), device=device, dtype=dtype)
    return x, y

def run_pytorch(*inputs):
    x, y = inputs
    return torch.dot(x, y).reshape(1)

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
