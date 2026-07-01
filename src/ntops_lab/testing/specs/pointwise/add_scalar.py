import torch
from ntops_lab.kernels.pointwise.add_scalar import run

OP_NAME = "add_scalar"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"
DEFAULT_NUMEL = 1048576

def make_inputs(numel=DEFAULT_NUMEL, device="cuda", dtype=torch.float32):
    x = torch.randn((numel,), device=device, dtype=dtype)
    y = torch.randn((numel,), device=device, dtype=dtype)
    return x, y

def run_pytorch(*inputs):
    x, y = inputs
    return torch.add(x, 1.25)

def check(atol=1.0e-3, rtol=1.0e-3):
    inputs = make_inputs()
    actual = run(*inputs)
    expected = run_pytorch(*inputs)
    torch.cuda.synchronize()
    if actual.dtype == torch.bool or expected.dtype == torch.bool or not torch.is_floating_point(expected):
        passed = torch.equal(actual, expected)
        if passed:
            max_abs_error = 0.0
        else:
            max_abs_error = (actual.to(torch.float32) - expected.to(torch.float32)).abs().max().item()
    else:
        max_abs_error = (actual - expected).abs().max().item()
        passed = torch.allclose(actual, expected, atol=atol, rtol=rtol)
    print(OP_NAME, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max_abs_error)
    if not passed:
        raise AssertionError(OP_NAME)
