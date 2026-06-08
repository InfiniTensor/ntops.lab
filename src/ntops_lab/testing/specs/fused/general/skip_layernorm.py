import torch
from ntops_lab.kernels.fused.general.skip_layernorm import run, HIDDEN

EPS = 1.0e-5
ROWS = 4096
OP_NAME = "skip_layernorm"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(device="cuda", dtype=torch.float32):
    x = torch.randn((ROWS, HIDDEN), device=device, dtype=dtype)
    skip = torch.randn((ROWS, HIDDEN), device=device, dtype=dtype)
    gamma = torch.randn((HIDDEN,), device=device, dtype=dtype)
    beta = torch.randn((HIDDEN,), device=device, dtype=dtype)
    return x, skip, gamma, beta

def run_pytorch(*inputs):
    x, skip, gamma, beta = inputs
    return torch.nn.functional.layer_norm(x + skip, (HIDDEN,), gamma, beta, EPS)

def check(atol=1.0e-4, rtol=1.0e-4):
    inputs = make_inputs()
    actual = run(*inputs)
    expected = run_pytorch(*inputs)
    torch.cuda.synchronize()
    max_abs_error = (actual - expected).abs().max().item()
    passed = torch.allclose(actual, expected, atol=atol, rtol=rtol)
    print(OP_NAME, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max_abs_error)
    if not passed:
        raise AssertionError(OP_NAME)
