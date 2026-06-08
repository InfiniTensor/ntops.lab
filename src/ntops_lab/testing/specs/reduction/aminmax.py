import torch
from ntops_lab.kernels.reduction.aminmax import run

OP_NAME = "aminmax"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(rows=32768, dim=32, device="cuda", dtype=torch.float32):
    x = torch.randn((rows, dim), device=device, dtype=dtype)
    return (x,)

def run_pytorch(*inputs):
    x, = inputs
    return x.min(dim=1).values, x.max(dim=1).values

def check(atol=1.0e-3, rtol=1.0e-3):
    inputs = make_inputs()
    actual = run(*inputs)
    expected = run_pytorch(*inputs)
    torch.cuda.synchronize()
    errors = [(a - e).abs().max().item() for a, e in zip(actual, expected)]
    passed = all(torch.allclose(a, e, atol=atol, rtol=rtol) for a, e in zip(actual, expected))
    print(OP_NAME, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max(errors))
    if not passed:
        raise AssertionError(OP_NAME)
