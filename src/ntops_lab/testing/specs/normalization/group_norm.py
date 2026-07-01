import torch
import torch.nn.functional as F
from ntops_lab.kernels.normalization.group_norm import run

OP_NAME = "group_norm"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"
TEST_DIMS = (64, 32)

def make_inputs(batch=64, channels=4, dim=64, device="cuda", dtype=torch.float32):
    x = torch.randn((batch, channels, dim), device=device, dtype=dtype) * 0.5
    return (x,)

def run_pytorch(*inputs):
    x, = inputs
    return F.group_norm(x, num_groups=4, weight=None, bias=None, eps=1.0e-5)

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
