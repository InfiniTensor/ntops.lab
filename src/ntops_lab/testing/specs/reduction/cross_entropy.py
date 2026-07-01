import torch
import torch.nn.functional as F
from ntops_lab.kernels.reduction.cross_entropy import run

OP_NAME = "cross_entropy"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"
CLASSES = (4, 7)


def make_inputs(rows=8192, classes=4, device="cuda", dtype=torch.float32):
    logits = torch.randn((rows, classes), device=device, dtype=dtype) * 0.5
    x = logits
    target = torch.randint(0, classes, (rows,), device=device, dtype=torch.int64)
    return x, target


def run_pytorch(*inputs):
    x, target = inputs
    return F.cross_entropy(x, target, reduction="none")


def check(atol=1.0e-3, rtol=1.0e-3):
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
