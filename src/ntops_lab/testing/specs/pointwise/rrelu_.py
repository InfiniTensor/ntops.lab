import torch
import torch.nn.functional as F
from ntops_lab.kernels.pointwise.rrelu_ import run

OP_NAME = "rrelu_"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(numel=1048576, device="cuda", dtype=torch.float32):
    return (torch.randn((numel,), device=device, dtype=dtype),)

def run_pytorch(*inputs):
    x = inputs[0].clone()
    return F.rrelu(x, lower=1.0 / 8.0, upper=1.0 / 3.0, training=False, inplace=True)

def check(atol=1.0e-3, rtol=1.0e-3):
    inputs = make_inputs()
    actual = run(*(item.clone() for item in inputs))
    expected = run_pytorch(*inputs)
    torch.cuda.synchronize()
    max_abs_error = (actual - expected).abs().max().item()
    passed = torch.allclose(actual, expected, atol=atol, rtol=rtol)
    print(OP_NAME, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max_abs_error)
    if not passed:
        raise AssertionError(OP_NAME)
