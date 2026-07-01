import torch
from ntops_lab.kernels.fused.general.gate_softshrink_mul import run

OP_NAME = "gate_softshrink_mul"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"
DEFAULT_NUMEL = 1048576

def make_inputs(numel=DEFAULT_NUMEL, device="cuda", dtype=torch.float32):
    a = torch.randn((numel,), device=device, dtype=dtype) * 0.5
    b = torch.randn((numel,), device=device, dtype=dtype) * 0.5
    return a, b

def run_pytorch(*inputs):
    a, b = inputs
    value = a
    return (torch.nn.functional.softshrink(value)) * b

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
