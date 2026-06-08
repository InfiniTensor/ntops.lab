import torch
import torch.nn.functional as F
from ntops_lab.kernels.layout.avg_pool2d import run, _compare

OP_NAME = "avg_pool2d"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(device="cuda", dtype=torch.float16):
    if OP_NAME in {"avg_pool3d", "max_pool3d_with_indices"}:
        return (torch.randn((1, 2, 5, 8, 8), device=device, dtype=dtype),)
    return (torch.randn((1, 3, 32, 32), device=device, dtype=dtype),)

def run_pytorch(*inputs):
    (x,) = inputs
    if OP_NAME == "avg_pool2d":
        return F.avg_pool2d(x, kernel_size=4, stride=2, padding=0)
    if OP_NAME == "max_pool2d_with_indices":
        return F.max_pool2d(x, kernel_size=4, stride=2, padding=0, return_indices=True)
    if OP_NAME == "avg_pool3d":
        return F.avg_pool3d(x, kernel_size=(2, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))
    if OP_NAME == "max_pool3d_with_indices":
        return F.max_pool3d(x, kernel_size=(2, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0), return_indices=True)
    raise NotImplementedError(OP_NAME)

def check(atol=1.0e-1, rtol=1.0e-1):
    inputs = make_inputs()
    actual = run(*inputs)
    expected = run_pytorch(*inputs)
    torch.cuda.synchronize()
    passed, max_abs_error = _compare(actual, expected, atol, rtol)
    print(OP_NAME, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max_abs_error)
    if not passed:
        raise AssertionError(OP_NAME)
