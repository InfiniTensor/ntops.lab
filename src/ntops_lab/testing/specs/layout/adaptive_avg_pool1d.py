import torch
import torch.nn.functional as F
from ntops_lab.kernels.layout.adaptive_avg_pool1d import run

OP_NAME = "adaptive_avg_pool1d"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(device="cuda", dtype=torch.float32):
    return (torch.randn((2, 3, 64), device=device, dtype=dtype) * 0.5,)

def run_pytorch(*inputs):
    (x,) = inputs
    return F.adaptive_avg_pool1d(x, 16)

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
