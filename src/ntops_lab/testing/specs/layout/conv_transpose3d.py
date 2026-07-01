import torch
import torch.nn.functional as F
from ntops_lab.kernels.layout.conv_transpose3d import run

OP_NAME = "conv_transpose3d"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(device="cuda", dtype=torch.float16):
    x = torch.randn((1, 2, 4, 5, 5), device=device, dtype=dtype) * 0.5
    w = torch.randn((2, 3, 3, 3, 3), device=device, dtype=dtype) * 0.5
    b = torch.randn((3,), device=device, dtype=dtype) * 0.5
    return x, w, b

def run_pytorch(*inputs):
    x, w, b = inputs
    return F.conv_transpose3d(x, w, b)

def check(atol=1.0e-1, rtol=1.0e-1):
    inputs = make_inputs()
    actual = run(*inputs)
    expected = run_pytorch(*inputs)
    torch.cuda.synchronize()
    max_abs_error = (actual - expected).abs().max().item()
    passed = torch.allclose(actual, expected, atol=atol, rtol=rtol)
    print(OP_NAME, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max_abs_error)
    if not passed:
        raise AssertionError(OP_NAME)
