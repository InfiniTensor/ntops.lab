import torch
import torch.nn.functional as F
from ntops_lab.kernels.linear.linear import run

OP_NAME = "linear"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(m=32, n=32, k=32, device="cuda", dtype=torch.float16):
    x = torch.randn((m, k), device=device, dtype=dtype)
    weight = torch.randn((n, k), device=device, dtype=dtype)
    bias = torch.randn((n,), device=device, dtype=dtype)
    return x, weight, bias

def run_pytorch(*inputs):
    x, weight, bias = inputs
    return F.linear(x, weight, bias)

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
