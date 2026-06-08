import torch
import torch.nn.functional as F
from ntops_lab.kernels.layout.conv_depthwise2d import run

OP_NAME = "conv_depthwise2d"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(device="cuda", dtype=torch.float16):
    if OP_NAME == "conv1d":
        x = torch.randn((1, 2, 16), device=device, dtype=dtype)
        w = torch.randn((3, 2, 3), device=device, dtype=dtype)
        b = torch.randn((3,), device=device, dtype=dtype)
        return x, w, b
    if OP_NAME == "conv3d":
        x = torch.randn((1, 2, 5, 8, 8), device=device, dtype=dtype)
        w = torch.randn((3, 2, 3, 3, 3), device=device, dtype=dtype)
        b = torch.randn((3,), device=device, dtype=dtype)
        return x, w, b
    if OP_NAME == "conv_depthwise2d":
        x = torch.randn((1, 4, 8, 8), device=device, dtype=dtype)
        w = torch.randn((4, 1, 3, 3), device=device, dtype=dtype)
        b = torch.randn((4,), device=device, dtype=dtype)
        return x, w, b
    if OP_NAME == "conv_transpose1d":
        x = torch.randn((1, 2, 8), device=device, dtype=dtype)
        w = torch.randn((2, 3, 3), device=device, dtype=dtype)
        b = torch.randn((3,), device=device, dtype=dtype)
        return x, w, b
    if OP_NAME == "conv_transpose2d":
        x = torch.randn((1, 2, 8, 8), device=device, dtype=dtype)
        w = torch.randn((2, 3, 3, 3), device=device, dtype=dtype)
        b = torch.randn((3,), device=device, dtype=dtype)
        return x, w, b
    x = torch.randn((1, 2, 8, 8), device=device, dtype=dtype)
    w = torch.randn((3, 2, 3, 3), device=device, dtype=dtype)
    b = torch.randn((3,), device=device, dtype=dtype)
    return x, w, b

def run_pytorch(*inputs):
    x, w, b = inputs
    if OP_NAME == "conv1d":
        return F.conv1d(x, w, b, padding=1)
    if OP_NAME == "conv3d":
        return F.conv3d(x, w, b)
    if OP_NAME == "conv_depthwise2d":
        return F.conv2d(x, w, b, padding=1, groups=x.shape[1])
    if OP_NAME == "conv_transpose1d":
        return F.conv_transpose1d(x, w, b)
    if OP_NAME == "conv_transpose2d":
        return F.conv_transpose2d(x, w, b)
    return F.conv2d(x, w, b, padding=1)

def check(atol=1.0e-1, rtol=1.0e-1):
    inputs = make_inputs()
    actual = run(*inputs)
    expected = run_pytorch(*inputs)
    torch.cuda.synchronize()
    max_abs_error = (actual - expected).abs().max().item()
    passed = torch.allclose(actual, expected, atol=atol, rtol=rtol)
    print(OP_NAME, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max_abs_error)
    if not passed:
        raise AssertionError(OP_NAME)
