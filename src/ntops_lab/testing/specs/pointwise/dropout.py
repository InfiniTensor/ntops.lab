import torch
import torch.nn.functional as F
from ntops_lab.kernels.pointwise.dropout import run

OP_NAME = "dropout"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(device="cuda", dtype=torch.float32):
    return (torch.randn((1024, 1024), device=device, dtype=dtype) * 0.5,)

def run_pytorch(*inputs):
    (x,) = inputs
    return F.dropout(x, p=0.5, training=False)

def check(atol=1.0e-6, rtol=1.0e-6):
    inputs = make_inputs()
    actual = run(*inputs)
    expected = run_pytorch(*inputs)
    torch.cuda.synchronize()
    if actual.dtype == torch.bool or not torch.is_floating_point(expected):
        passed = torch.equal(actual, expected)
        max_abs_error = 0.0 if passed else (actual.to(torch.float32) - expected.to(torch.float32)).abs().max().item()
    else:
        max_abs_error = (actual - expected).abs().max().item()
        passed = torch.allclose(actual, expected, atol=atol, rtol=rtol)
    print(OP_NAME, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max_abs_error)
    if not passed:
        raise AssertionError(OP_NAME)
