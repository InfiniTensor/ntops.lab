import torch
import torch.nn.functional as F
from ntops_lab.kernels.fused.general.multilabel_soft_margin_loss import run

OP_NAME = "multilabel_soft_margin_loss"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"
TEST_DIMS = (32, 64)

def make_inputs(rows=8192, dim=32, device="cuda", dtype=torch.float32):
    x = torch.randn((rows, dim), device=device, dtype=dtype) * 0.5
    target = torch.rand((rows, dim), device=device, dtype=dtype)
    return x, target

def run_pytorch(*inputs):
    x, target = inputs
    return F.multilabel_soft_margin_loss(x, target, reduction="none")

def check(atol=1.0e-3, rtol=1.0e-3):
    for dim in TEST_DIMS:
        inputs = make_inputs(dim=dim)
        actual = run(*inputs)
        expected = run_pytorch(*inputs)
        torch.cuda.synchronize()
        max_abs_error = (actual - expected).abs().max().item()
        passed = torch.allclose(actual, expected, atol=atol, rtol=rtol)
        print(OP_NAME, "dim=", dim, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max_abs_error)
        if not passed:
            raise AssertionError(OP_NAME)
