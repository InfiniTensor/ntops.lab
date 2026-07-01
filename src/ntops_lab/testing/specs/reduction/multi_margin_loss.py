import torch
import torch.nn.functional as F
from ntops_lab.kernels.reduction.multi_margin_loss import run

OP_NAME = "multi_margin_loss"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"
CLASSES = (16, 10)


def make_inputs(classes=16, rows=8192, device="cuda", dtype=torch.float32):
    x = torch.randn((rows, classes), device=device, dtype=dtype) * 0.5
    target = torch.randint(0, classes, (rows,), device=device, dtype=torch.long)
    return x, target


def run_pytorch(*inputs):
    x, target = inputs
    return F.multi_margin_loss(x, target, p=1, margin=1.0, weight=None, reduction="none")


def check(atol=1.0e-5, rtol=1.0e-5):
    for classes in CLASSES:
        inputs = make_inputs(classes=classes)
        actual = run(*inputs)
        expected = run_pytorch(*inputs)
        torch.cuda.synchronize()
        max_abs_error = (actual - expected).abs().max().item()
        passed = torch.allclose(actual, expected, atol=atol, rtol=rtol)
        print(OP_NAME, "classes=", classes, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max_abs_error)
        if not passed:
            raise AssertionError(OP_NAME)
