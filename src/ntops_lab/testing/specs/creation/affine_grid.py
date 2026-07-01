import torch
import torch.nn.functional as F
from ntops_lab.kernels.creation.affine_grid import run

OP_NAME = "affine_grid"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"
SIZES = ((16, 16), (12, 20))


def make_inputs(size, device="cuda", dtype=torch.float32):
    theta = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.8, 0.1, 0.2], [-0.1, 0.9, -0.15]],
            [[1.1, -0.2, 0.05], [0.2, 0.7, 0.1]],
            [[0.6, 0.3, -0.25], [-0.3, 1.0, 0.25]],
        ],
        device=device,
        dtype=dtype,
    )
    return theta, (theta.shape[0], 3, size[0], size[1])


def run_pytorch(*inputs):
    theta, size = inputs
    return F.affine_grid(theta, size=size, align_corners=False)


def check(atol=1.0e-6, rtol=1.0e-6):
    for size in SIZES:
        inputs = make_inputs(size)
        actual = run(*inputs)
        expected = run_pytorch(*inputs)
        torch.cuda.synchronize()
        max_abs_error = (actual - expected).abs().max().item()
        passed = torch.allclose(actual, expected, atol=atol, rtol=rtol)
        print(OP_NAME, "size=", size, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max_abs_error)
        if not passed:
            raise AssertionError(OP_NAME)
