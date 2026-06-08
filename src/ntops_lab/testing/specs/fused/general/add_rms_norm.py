import torch
from ntops_lab.kernels.fused.general.add_rms_norm import run, HIDDEN

EPS = 1.0e-5
ROWS = 4096
OP_NAME = "add_rms_norm"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(device="cuda", dtype=torch.float32):
    x = torch.randn((ROWS, HIDDEN), device=device, dtype=dtype)
    residual = torch.randn((ROWS, HIDDEN), device=device, dtype=dtype)
    weight = torch.randn((HIDDEN,), device=device, dtype=dtype)
    return x, residual, weight

def run_pytorch(*inputs):
    x, residual, weight = inputs
    y = x + residual
    return y * torch.rsqrt(torch.mean(y * y, dim=1, keepdim=True) + EPS) * weight

def check(atol=1.0e-4, rtol=1.0e-4):
    inputs = make_inputs()
    actual = run(*inputs)
    expected = run_pytorch(*inputs)
    torch.cuda.synchronize()
    max_abs_error = (actual - expected).abs().max().item()
    passed = torch.allclose(actual, expected, atol=atol, rtol=rtol)
    print(OP_NAME, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max_abs_error)
    if not passed:
        raise AssertionError(OP_NAME)
