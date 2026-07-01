import torch
from ntops_lab.kernels.fused.general.rms_norm_hardshrink import run

EPS = 1.0e-5
DEFAULT_HIDDEN = 32
TEST_HIDDENS = (32, 64)
ROWS = 2048
OP_NAME = "rms_norm_hardshrink"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(device="cuda", dtype=torch.float32, hidden=DEFAULT_HIDDEN):
    x = torch.randn((ROWS, hidden), device=device, dtype=dtype) * 0.5
    weight = torch.randn((hidden,), device=device, dtype=dtype) * 0.5
    return x, weight

def run_pytorch(*inputs):
    x, weight = inputs
    value = x * torch.rsqrt((x * x).mean(dim=1, keepdim=True) + EPS) * weight
    return torch.nn.functional.hardshrink(value)

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
