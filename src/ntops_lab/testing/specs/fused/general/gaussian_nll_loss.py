import torch
import torch.nn.functional as F
from ntops_lab.kernels.fused.general.gaussian_nll_loss import run

OP_NAME = "gaussian_nll_loss"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(numel=1048576, device="cuda", dtype=torch.float32):
    x = torch.randn((numel,), device=device, dtype=dtype) * 0.5
    target = torch.randn((numel,), device=device, dtype=dtype) * 0.5
    var = torch.rand((numel,), device=device, dtype=dtype) + 0.5
    return x, target, var

def run_pytorch(*inputs):
    x, target, var = inputs
    return F.gaussian_nll_loss(x, target, var, reduction="none")

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
