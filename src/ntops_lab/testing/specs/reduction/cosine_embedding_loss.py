import torch
import torch.nn.functional as F
from ntops_lab.kernels.reduction.cosine_embedding_loss import run

OP_NAME = "cosine_embedding_loss"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"


TEST_DIMS = (64, 32)

def make_inputs(rows=8192, dim=64, device="cuda", dtype=torch.float32):
    x1 = torch.randn((rows, dim), device=device, dtype=dtype) * 0.5
    x2 = torch.randn((rows, dim), device=device, dtype=dtype) * 0.5
    label = torch.where(torch.rand((rows,), device=device) > 0.5, 1.0, -1.0).to(dtype)
    return x1, x2, label

def run_pytorch(*inputs):
    x1, x2, label = inputs
    return F.cosine_embedding_loss(x1, x2, label, margin=0.0, reduction="none")

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
