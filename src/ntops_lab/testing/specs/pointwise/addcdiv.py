import torch
from ntops_lab.kernels.pointwise.addcdiv import run

OP_NAME = "addcdiv"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"
DEFAULT_NUMEL = 1048576

def make_inputs(numel=DEFAULT_NUMEL, device="cuda", dtype=torch.float32):
    x = torch.randn((numel,), device=device, dtype=dtype)
    y = torch.randn((numel,), device=device, dtype=dtype)
    z = torch.randn((numel,), device=device, dtype=dtype) + 1.25
    return x, y, z

def run_pytorch(*inputs):
    x, y, z = inputs
    return torch.addcdiv(x, y, z, value=1.0)

def check(atol=1.0e-3, rtol=1.0e-3):
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
