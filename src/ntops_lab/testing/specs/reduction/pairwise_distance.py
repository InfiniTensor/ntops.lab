import torch
import torch.nn.functional as F
from ntops_lab.kernels.reduction.pairwise_distance import run

OP_NAME = "pairwise_distance"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"


TEST_DIMS = (64, 32)

def make_inputs(rows=8192, dim=64, device="cuda", dtype=torch.float32):
    x = torch.randn((rows, dim), device=device, dtype=dtype) * 0.5
    y = torch.randn((rows, dim), device=device, dtype=dtype) * 0.5
    return x, y

def run_pytorch(*inputs):
    x, y = inputs
    return F.pairwise_distance(x, y, p=2.0, eps=1.0e-6)

def check(atol=1.0e-3, rtol=1.0e-3):
    for dim in TEST_DIMS:
        inputs = make_inputs(dim=dim)
        actual = run(*inputs)
        expected = run_pytorch(*inputs)
        torch.cuda.synchronize()
        if actual.dtype == torch.bool or expected.dtype == torch.bool:
            passed = torch.equal(actual.bool(), expected.bool())
            max_abs_error = 0.0
        else:
            max_abs_error = (actual - expected).abs().max().item()
            passed = torch.allclose(actual, expected, atol=atol, rtol=rtol)
        print(OP_NAME, "dim=", dim, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max_abs_error)
        if not passed:
            raise AssertionError(OP_NAME)
