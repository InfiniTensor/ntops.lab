import torch

from ntops_lab.kernels.linear.baddbmm import run

OP_NAME = "baddbmm"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(batch=4, m=32, n=32, k=32, device="cuda", dtype=torch.float16):
    c = torch.randn((batch, m, n), device=device, dtype=dtype)
    a = torch.randn((batch, m, k), device=device, dtype=dtype)
    b = torch.randn((batch, k, n), device=device, dtype=dtype)
    return c, a, b

def run_pytorch(*inputs):
    c, a, b = inputs
    return torch.baddbmm(c, a, b)

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
