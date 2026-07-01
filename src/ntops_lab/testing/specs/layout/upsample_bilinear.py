import torch
import torch.nn.functional as F
from ntops_lab.kernels.layout.upsample_bilinear import run

OP_NAME = "upsample_bilinear"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"
TEST_SHAPES = ((8, 3, 16, 16), (4, 2, 12, 20))


def make_inputs(shape, device="cuda", dtype=torch.float32):
    return (torch.randn(shape, device=device, dtype=dtype) * 0.5,)


def run_pytorch(*inputs, scale_factor=2):
    (x,) = inputs
    return F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=True)


def check(atol=1.0e-5, rtol=1.0e-5):
    for shape in TEST_SHAPES:
        inputs = make_inputs(shape)
        actual = run(*inputs, scale_factor=2)
        expected = run_pytorch(*inputs, scale_factor=2)
        torch.cuda.synchronize()
        max_abs_error = (actual - expected).abs().max().item()
        passed = torch.allclose(actual, expected, atol=atol, rtol=rtol)
        print(OP_NAME, "shape=", shape, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max_abs_error)
        if not passed:
            raise AssertionError(OP_NAME)
