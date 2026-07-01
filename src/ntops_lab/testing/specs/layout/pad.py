import torch
import torch.nn.functional as F
from ntops_lab.kernels.layout.pad import run

OP_NAME = "pad"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(device="cuda", dtype=torch.float32):
    return (torch.randn((512, 512), device=device, dtype=dtype) * 0.5,)

def run_pytorch(*inputs):
    x, = inputs
    return F.pad(x, (2, 3, 1, 4), mode="constant", value=0.0)

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
