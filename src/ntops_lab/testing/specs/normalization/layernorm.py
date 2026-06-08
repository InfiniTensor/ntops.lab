import torch
from ntops_lab.kernels.normalization.layernorm import run

DEFAULT_DIM = 32
DEFAULT_ROWS = 32768
OP_NAME = "layernorm"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(rows=DEFAULT_ROWS, dim=DEFAULT_DIM, device="cuda", dtype=torch.float32):
    x = torch.randn((rows, dim), device=device, dtype=dtype)
    weight = torch.randn((dim,), device=device, dtype=dtype)
    bias = torch.randn((dim,), device=device, dtype=dtype)
    return x, weight, bias

def run_pytorch(*inputs):
    x, weight, bias = inputs
    return torch.nn.functional.layer_norm(x, (x.shape[1],), weight, bias, eps=1.0e-5)

def check(atol=1.0e-3, rtol=1.0e-3):
    inputs = make_inputs()
    actual = run(*inputs)
    expected = run_pytorch(*inputs)
    torch.cuda.synchronize()
    max_abs_error = (actual - expected).abs().max().item()
    passed = torch.allclose(actual, expected, atol=atol, rtol=rtol)
    print(OP_NAME, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max_abs_error)
    if not passed:
        raise AssertionError(OP_NAME)
