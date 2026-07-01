import torch
import torch.nn.functional as F
from ntops_lab.kernels.fused.general.margin_ranking_loss import run

OP_NAME = "margin_ranking_loss"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(numel=1048576, device="cuda", dtype=torch.float32):
    x1 = torch.randn((numel,), device=device, dtype=dtype) * 0.5
    x2 = torch.randn((numel,), device=device, dtype=dtype) * 0.5
    label = torch.where(torch.rand((numel,), device=device) > 0.5, 1.0, -1.0).to(dtype)
    return x1, x2, label

def run_pytorch(*inputs):
    x1, x2, label = inputs
    return F.margin_ranking_loss(x1, x2, label, margin=1.0, reduction="none")

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
