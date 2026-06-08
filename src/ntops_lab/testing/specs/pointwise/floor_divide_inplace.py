import torch
from ntops_lab.kernels.pointwise.floor_divide_inplace import run

OP_NAME = "floor_divide_"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(device="cuda", dtype=torch.float32):
    x = torch.rand((1048576,), device=device, dtype=dtype) * 0.5 + 0.1
    y = torch.rand((1048576,), device=device, dtype=dtype) * 16.0 + 1.0
    return x, y

def run_pytorch(*inputs):
    x, y = inputs
    return torch.div(x, y, rounding_mode='floor')

def check(atol=1.0e-4, rtol=1.0e-4):
    inputs = make_inputs()
    actual = run(*inputs)
    expected = run_pytorch(*inputs)
    torch.cuda.synchronize()
    if actual.dtype == torch.bool:
        passed = torch.equal(actual, expected)
        max_abs_error = 0.0 if passed else 1.0
    else:
        max_abs_error = (actual - expected).abs().max().item()
        passed = torch.allclose(actual, expected, atol=atol, rtol=rtol)
    print(OP_NAME, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max_abs_error)
    if not passed:
        raise AssertionError(OP_NAME)
