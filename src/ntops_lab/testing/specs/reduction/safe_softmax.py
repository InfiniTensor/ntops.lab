import torch
from ntops_lab.kernels.reduction.safe_softmax import run

DEFAULT_DIM = 32
DEFAULT_ROWS = 32768
OP_NAME = "_safe_softmax"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(rows=DEFAULT_ROWS, dim=DEFAULT_DIM, device="cuda", dtype=torch.float32):
    x = torch.randn((rows, dim), device=device, dtype=dtype)
    return (x,)

def run_pytorch(*inputs):
    x, = inputs
    return torch.softmax(x, dim=1)

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
