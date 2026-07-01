import torch
from ntops_lab.kernels.creation.eye_m import run

OP_NAME = "eye_m"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"
SHAPES = ((512, None), (257, 129))


def make_inputs(device="cuda", dtype=torch.float32):
    return ()


def run_pytorch(*inputs, n=512, m=None, device="cuda", dtype=torch.float32):
    if m is None:
        return torch.eye(n, device=device, dtype=dtype)
    return torch.eye(n, m, device=device, dtype=dtype)


def check(atol=0.0, rtol=0.0):
    for n, m in SHAPES:
        actual = run(n=n, m=m)
        expected = run_pytorch(n=n, m=m)
        torch.cuda.synchronize()
        max_abs_error = (actual - expected).abs().max().item()
        passed = torch.equal(actual, expected)
        print(OP_NAME, "n=", n, "m=", m, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max_abs_error)
        if not passed:
            raise AssertionError(OP_NAME)
