import torch
import torch.nn.functional as F
from ntops_lab.kernels.layout.fractional_max_pool2d import run

OP_NAME = "fractional_max_pool2d"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(device="cuda", dtype=torch.float32):
    x = torch.randn((16, 32, 64, 64), device=device, dtype=dtype)
    random_samples = torch.zeros((16, 32, 2), device=device, dtype=dtype)
    return x, random_samples

def run_pytorch(*inputs):
    x, random_samples = inputs
    return F.fractional_max_pool2d(x, 2, output_size=(x.shape[-2] // 2, x.shape[-1] // 2), _random_samples=random_samples)

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
