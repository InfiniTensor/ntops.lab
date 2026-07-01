import torch
import torch.nn.functional as F
from ntops_lab.kernels.normalization.batch_norm import run

OP_NAME = "batch_norm"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(rows=8192, channels=64, device="cuda", dtype=torch.float32):
    x = torch.randn((rows, channels), device=device, dtype=dtype) * 0.5
    running_mean = torch.randn((channels,), device=device, dtype=dtype) * 0.1
    running_var = torch.rand((channels,), device=device, dtype=dtype) + 0.5
    weight = torch.randn((channels,), device=device, dtype=dtype) * 0.5
    bias = torch.randn((channels,), device=device, dtype=dtype) * 0.5
    return x, running_mean, running_var, weight, bias

def run_pytorch(*inputs):
    x, running_mean, running_var, weight, bias = inputs
    return F.batch_norm(x, running_mean, running_var, weight, bias, training=False, momentum=0.1, eps=1.0e-5)

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
