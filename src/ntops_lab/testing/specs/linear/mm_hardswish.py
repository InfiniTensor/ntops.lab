import torch
from ntops_lab.kernels.linear.mm_hardswish import run

OP_NAME = "mm_hardswish"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(m=32, n=32, k=32, device="cuda", dtype=torch.float16):
    a = torch.randn((m, k), device=device, dtype=dtype) * 0.5
    b = torch.randn((k, n), device=device, dtype=dtype) * 0.5
    return a, b

def run_pytorch(*inputs):
    a, b = inputs
    return torch.nn.functional.hardswish(a @ b)

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
