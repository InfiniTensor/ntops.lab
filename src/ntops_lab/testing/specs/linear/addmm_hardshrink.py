import torch
from ntops_lab.kernels.linear.addmm_hardshrink import run

OP_NAME = "addmm_hardshrink"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(m=32, n=32, k=32, device="cuda", dtype=torch.float16):
    c = torch.randn((m, n), device=device, dtype=dtype) * 0.5
    a = torch.randn((m, k), device=device, dtype=dtype) * 0.5
    b = torch.randn((k, n), device=device, dtype=dtype) * 0.5
    return c, a, b

def run_pytorch(*inputs):
    c, a, b = inputs
    value = c + a @ b
    return torch.nn.functional.hardshrink(value)

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
