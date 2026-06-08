import torch
from ntops_lab.kernels.creation.eye_m import run, N

OP_NAME = "eye_m"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(device="cuda", dtype=torch.float32):
    return ()

def run_pytorch(*inputs, device="cuda", dtype=torch.float32):
    return torch.eye(N, device=device, dtype=dtype)

def check(atol=0.0, rtol=0.0):
    actual = run()
    expected = run_pytorch()
    torch.cuda.synchronize()
    max_abs_error = (actual - expected).abs().max().item()
    passed = torch.equal(actual, expected)
    print(OP_NAME, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max_abs_error)
    if not passed:
        raise AssertionError(OP_NAME)
