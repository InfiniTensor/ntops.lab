import torch
from ntops_lab.kernels.fused.general.gate_hardswish_mul_add import run

OP_NAME = "gate_hardswish_mul_add"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"
DEFAULT_NUMEL = 1048576

def make_inputs(numel=DEFAULT_NUMEL, device="cuda", dtype=torch.float32):
    a = torch.randn((numel,), device=device, dtype=dtype) * 0.5
    b = torch.randn((numel,), device=device, dtype=dtype) * 0.5
    c = torch.randn((numel,), device=device, dtype=dtype) * 0.5
    return a, b, c

def run_pytorch(*inputs):
    a, b, c = inputs
    return torch.nn.functional.hardswish(a) * b + c

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
