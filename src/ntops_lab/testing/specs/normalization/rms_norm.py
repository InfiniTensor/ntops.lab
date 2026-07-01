import torch
from ntops_lab.kernels.normalization.rms_norm import run

DEFAULT_DIM = 32
TEST_DIMS = (32, 64)
DEFAULT_ROWS = 32768
OP_NAME = "rms_norm"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(rows=DEFAULT_ROWS, dim=DEFAULT_DIM, device="cuda", dtype=torch.float32):
    x = torch.randn((rows, dim), device=device, dtype=dtype)
    weight = torch.randn((dim,), device=device, dtype=dtype)
    return x, weight

def run_pytorch(*inputs):
    x, weight = inputs
    return x * torch.rsqrt(x.square().mean(dim=1, keepdim=True) + 1.0e-5) * weight

def check(atol=1.0e-3, rtol=1.0e-3):
    for dim in TEST_DIMS:
        inputs = make_inputs(dim=dim)
        actual = run(*inputs)
        expected = run_pytorch(*inputs)
        torch.cuda.synchronize()
        max_abs_error = (actual - expected).abs().max().item()
        passed = torch.allclose(actual, expected, atol=atol, rtol=rtol)
        print(OP_NAME, "dim=", dim, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max_abs_error)
        if not passed:
            raise AssertionError(f"{OP_NAME} dim={dim}")
