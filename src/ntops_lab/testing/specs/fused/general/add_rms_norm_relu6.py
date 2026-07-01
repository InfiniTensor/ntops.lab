import torch
from ntops_lab.kernels.fused.general.add_rms_norm_relu6 import run

EPS = 1.0e-5
DEFAULT_HIDDEN = 32
TEST_HIDDENS = (32, 64)
ROWS = 2048
OP_NAME = "add_rms_norm_relu6"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(device="cuda", dtype=torch.float32, hidden=DEFAULT_HIDDEN):
    x = torch.randn((ROWS, hidden), device=device, dtype=dtype) * 0.5
    residual = torch.randn((ROWS, hidden), device=device, dtype=dtype) * 0.5
    weight = torch.randn((hidden,), device=device, dtype=dtype) * 0.5
    return x, residual, weight

def run_pytorch(*inputs):
    x, residual, weight = inputs
    value = (x + residual) * torch.rsqrt(((x + residual) * (x + residual)).mean(dim=1, keepdim=True) + EPS) * weight
    return torch.clamp(value, 0.0, 6.0)

def check(atol=1.0e-4, rtol=1.0e-4):
    for hidden in TEST_HIDDENS:
        inputs = make_inputs(hidden=hidden)
        actual = run(*inputs)
        expected = run_pytorch(*inputs)
        torch.cuda.synchronize()
        max_abs_error = (actual - expected).abs().max().item()
        passed = torch.allclose(actual, expected, atol=atol, rtol=rtol)
        print(OP_NAME, "hidden=", hidden, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max_abs_error)
        if not passed:
            raise AssertionError(f"{OP_NAME} hidden={hidden}")
