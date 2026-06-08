import torch
from ntops_lab.kernels.pointwise.where_scalar_other import run

OP_NAME = "where_scalar_other"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(device="cuda", dtype=torch.float32):
    x = torch.randint(0, 2, (1048576,), device=device, dtype=torch.int32)
    y = torch.randn((1048576,), device=device, dtype=dtype)
    z = torch.randn((1048576,), device=device, dtype=dtype)
    return x, y, z

def run_pytorch(*inputs):
    x, y, z = inputs
    return torch.where(x != 0, y, z)

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
