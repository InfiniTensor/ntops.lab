import torch
import torch.nn.functional as F
from ntops_lab.kernels.fused.general.swiglu import run

OP_NAME = "swiglu"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(numel=1048576, device="cuda", dtype=torch.float32):
    a = torch.randn((numel,), device=device, dtype=dtype)
    b = torch.randn((numel,), device=device, dtype=dtype)
    return a, b

def run_pytorch(*inputs):
    a, b = inputs
    return F.silu(a) * b

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
