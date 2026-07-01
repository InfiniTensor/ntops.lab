import torch
import torch.nn.functional as F
from ntops_lab.kernels.layout.unfold import run

OP_NAME = "unfold"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(device="cuda", dtype=torch.float32):
    return (torch.randn((8, 3, 32, 32), device=device, dtype=dtype) * 0.5,)

def run_pytorch(*inputs):
    (x,) = inputs
    return F.unfold(x, kernel_size=(3, 3), dilation=1, padding=(1, 1), stride=(1, 1))

def check(atol=1.0e-6, rtol=1.0e-6):
    inputs = make_inputs()
    actual = run(*inputs)
    expected = run_pytorch(*inputs)
    torch.cuda.synchronize()
    max_abs_error = (actual - expected).abs().max().item()
    passed = torch.allclose(actual, expected, atol=atol, rtol=rtol)
    print(OP_NAME, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max_abs_error)
    if not passed:
        raise AssertionError(OP_NAME)
