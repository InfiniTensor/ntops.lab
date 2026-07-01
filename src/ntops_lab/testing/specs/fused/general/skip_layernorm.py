import torch
from ntops_lab.kernels.fused.general.skip_layernorm import run

EPS = 1.0e-5
DEFAULT_HIDDEN = 32
TEST_HIDDENS = (32, 64)
ROWS = 4096
OP_NAME = "skip_layernorm"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(device="cuda", dtype=torch.float32, hidden=DEFAULT_HIDDEN):
    x = torch.randn((ROWS, hidden), device=device, dtype=dtype)
    skip = torch.randn((ROWS, hidden), device=device, dtype=dtype)
    gamma = torch.randn((hidden,), device=device, dtype=dtype)
    beta = torch.randn((hidden,), device=device, dtype=dtype)
    return x, skip, gamma, beta

def run_pytorch(*inputs):
    x, skip, gamma, beta = inputs
    return torch.nn.functional.layer_norm(x + skip, (x.shape[1],), gamma, beta, EPS)

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
