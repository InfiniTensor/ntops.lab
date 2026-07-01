import torch
import torch.nn.functional as F
from ntops_lab.kernels.layout.fold import run

OP_NAME = "fold"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"
CASES = ((1, 4, 4), (2, 5, 6))


def make_inputs(channels, output_size, device="cuda", dtype=torch.float32):
    base = torch.randn((1, channels, output_size[0], output_size[1]), device=device, dtype=dtype) * 0.5
    return (F.unfold(base, kernel_size=(3, 3), dilation=1, padding=1, stride=1),)


def run_pytorch(*inputs, output_size):
    (x,) = inputs
    return F.fold(x, output_size=output_size, kernel_size=(3, 3), dilation=1, padding=1, stride=1)


def check(atol=1.0e-5, rtol=1.0e-5):
    for channels, out_h, out_w in CASES:
        output_size = (out_h, out_w)
        inputs = make_inputs(channels, output_size)
        actual = run(*inputs, output_size=output_size, kernel_size=(3, 3), dilation=1, padding=1, stride=1)
        expected = run_pytorch(*inputs, output_size=output_size)
        torch.cuda.synchronize()
        max_abs_error = (actual - expected).abs().max().item()
        passed = torch.allclose(actual, expected, atol=atol, rtol=rtol)
        print(OP_NAME, "channels=", channels, "output_size=", output_size, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max_abs_error)
        if not passed:
            raise AssertionError(OP_NAME)
