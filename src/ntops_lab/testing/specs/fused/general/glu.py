import torch
import torch.nn.functional as F
from ntops_lab.kernels.fused.general.glu import run

OP_NAME = "glu"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(rows=8192, dim=64, device="cuda", dtype=torch.float32):
    return (torch.randn((rows, dim), device=device, dtype=dtype) * 0.5,)

def run_pytorch(*inputs):
    x, = inputs
    return F.glu(x, dim=1)

def check(atol=1.0e-3, rtol=1.0e-3):
    inputs = make_inputs()
    actual = run(*inputs)
    expected = run_pytorch(*inputs)
    torch.cuda.synchronize()
    max_abs_error = (actual - expected).abs().max().item()
    passed = torch.allclose(actual, expected, atol=atol, rtol=rtol)
    print(OP_NAME, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max_abs_error)
    if not passed:
        raise AssertionError(OP_NAME)
