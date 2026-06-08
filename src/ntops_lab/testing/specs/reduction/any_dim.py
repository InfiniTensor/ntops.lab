import torch
from ntops_lab.kernels.reduction.any_dim import run, DIM

ROWS = 65536
OP_NAME = "any_dim"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(device="cuda", dtype=torch.float32):
    if OP_NAME.startswith(("all", "any")):
        x = torch.randint(0, 2, (ROWS, DIM), device=device, dtype=torch.int32).to(dtype)
    else:
        x = torch.randn((ROWS, DIM), device=device, dtype=dtype)
    return (x,)

def run_pytorch(*inputs):
    (x,) = inputs
    return torch.any(x.bool(), dim=1)

def check(atol=1.0e-4, rtol=1.0e-4):
    inputs = make_inputs()
    actual = run(*inputs)
    expected = run_pytorch(*inputs)
    torch.cuda.synchronize()
    if actual.dtype == torch.bool or expected.dtype == torch.bool:
        passed = torch.equal(actual.bool(), expected.bool())
        max_abs_error = 0.0
    else:
        max_abs_error = (actual - expected).abs().max().item()
        passed = torch.allclose(actual, expected, atol=atol, rtol=rtol)
    print(OP_NAME, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max_abs_error)
    if not passed:
        raise AssertionError(OP_NAME)
