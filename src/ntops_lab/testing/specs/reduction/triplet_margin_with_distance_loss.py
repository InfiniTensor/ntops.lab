import torch
import torch.nn.functional as F
from ntops_lab.kernels.reduction.triplet_margin_with_distance_loss import run

OP_NAME = "triplet_margin_with_distance_loss"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"


TEST_DIMS = (64, 32)

def make_inputs(rows=8192, dim=64, device="cuda", dtype=torch.float32):
    anchor = torch.randn((rows, dim), device=device, dtype=dtype) * 0.5
    positive = anchor + torch.randn((rows, dim), device=device, dtype=dtype) * 0.1
    negative = torch.randn((rows, dim), device=device, dtype=dtype) * 0.5
    return anchor, positive, negative

def run_pytorch(*inputs):
    anchor, positive, negative = inputs
    return F.triplet_margin_with_distance_loss(anchor, positive, negative, distance_function=lambda a, b: F.pairwise_distance(a, b, p=2.0, eps=1.0e-6), margin=1.0, swap=False, reduction="none")

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
