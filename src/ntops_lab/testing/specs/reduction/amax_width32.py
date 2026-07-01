import torch
from ntops_lab.kernels.reduction.amax_width32 import run

OP_NAME = "amax_width32"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"
DEFAULT_ROWS = 4096
DEFAULT_DIM = 32

TEST_DIMS = (32, 64)
def make_inputs(rows=DEFAULT_ROWS, dim=DEFAULT_DIM, device="cuda", dtype=torch.float32):
    x = torch.randn((rows, dim), device=device, dtype=dtype) * 0.5
    return (x,)

def run_pytorch(*inputs):
    x, = inputs
    return torch.amax(x, dim=1)

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
