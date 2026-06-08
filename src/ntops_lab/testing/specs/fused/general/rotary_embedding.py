import torch
from ntops_lab.kernels.fused.general.rotary_embedding import run

ROWS = 65536
OP_NAME = "rotary_embedding"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(device="cuda", dtype=torch.float32):
    x0 = torch.randn((ROWS,), device=device, dtype=dtype)
    x1 = torch.randn((ROWS,), device=device, dtype=dtype)
    angle = torch.randn((ROWS,), device=device, dtype=dtype)
    return x0, x1, torch.cos(angle), torch.sin(angle)

def run_pytorch(*inputs):
    x0, x1, cos, sin = inputs
    return x0 * cos - x1 * sin, x0 * sin + x1 * cos

def check(atol=1.0e-6, rtol=1.0e-6):
    inputs = make_inputs()
    actual = run(*inputs)
    expected = run_pytorch(*inputs)
    torch.cuda.synchronize()
    max_abs_error = max((actual[0] - expected[0]).abs().max().item(), (actual[1] - expected[1]).abs().max().item())
    passed = torch.allclose(actual[0], expected[0], atol=atol, rtol=rtol) and torch.allclose(actual[1], expected[1], atol=atol, rtol=rtol)
    print(OP_NAME, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max_abs_error)
    if not passed:
        raise AssertionError(OP_NAME)
