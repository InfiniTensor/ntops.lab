import torch
import torch.nn.functional as F
from ntops_lab.kernels.layout.upsample_nearest_exact1d import run

OP_NAME = "_upsample_nearest_exact1d"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"
SCALES = (2, 3)

def make_inputs(device="cuda", dtype=torch.float16):
    return (torch.randn((4, 8, 128), device=device, dtype=dtype),)

def run_pytorch(*inputs, scale_factor=2):
    (x,) = inputs
    return F.interpolate(x, scale_factor=scale_factor, mode="nearest")

def check(atol=0.0, rtol=0.0):
    inputs = make_inputs()
    passed = True
    max_abs_error = 0.0
    for scale in SCALES:
        actual = run(*inputs, scale_factor=scale)
        expected = run_pytorch(*inputs, scale_factor=scale)
        torch.cuda.synchronize()
        max_abs_error = max(max_abs_error, (actual - expected).abs().max().item())
        passed = passed and torch.equal(actual, expected)
    print(OP_NAME, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max_abs_error)
    if not passed:
        raise AssertionError(OP_NAME)
