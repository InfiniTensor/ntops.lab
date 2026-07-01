import torch
import torch.nn.functional as F
from ntops_lab.kernels.reduction.gumbel_softmax import SEED, run

OP_NAME = "gumbel_softmax"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(device="cuda", dtype=torch.float32, rows=8192):
    x = torch.randn((rows, 128), device=device, dtype=dtype) * 0.5
    return (x,)

def run_pytorch(*inputs):
    (x,) = inputs
    torch.manual_seed(SEED)
    return F.gumbel_softmax(x, tau=1.0, hard=False, dim=-1)

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
