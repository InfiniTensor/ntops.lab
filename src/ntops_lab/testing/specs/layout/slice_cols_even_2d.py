import torch
from ntops_lab.kernels.layout.slice_cols_even_2d import run

OP_NAME = "slice_cols_even_2d"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(device="cuda", dtype=torch.float32):
    x = torch.randn((16, 32), device=device, dtype=dtype)
    return (x,)

def run_pytorch(*inputs):
    x, = inputs
    return x[:, ::2].contiguous()

def check(atol=0.0, rtol=0.0):
    inputs = make_inputs()
    actual = run(*inputs)
    expected = run_pytorch(*inputs)
    torch.cuda.synchronize()
    max_abs_error = (actual - expected).abs().max().item()
    passed = torch.equal(actual, expected)
    print(OP_NAME, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max_abs_error)
    if not passed:
        raise AssertionError(OP_NAME)
