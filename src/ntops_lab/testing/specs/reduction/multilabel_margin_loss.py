import torch
import torch.nn.functional as F
from ntops_lab.kernels.reduction.multilabel_margin_loss import run

OP_NAME = "multilabel_margin_loss"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"
CLASSES = (16, 10)


def make_inputs(classes=16, rows=2048, labels_per_row=3, device="cuda", dtype=torch.float32):
    x = torch.randn((rows, classes), device=device, dtype=dtype) * 0.5
    target = torch.full((rows, classes), -1, device=device, dtype=torch.long)
    labels = torch.stack([torch.randperm(classes, device=device)[:labels_per_row] for _ in range(rows)])
    target[:, :labels_per_row] = labels
    return x, target


def run_pytorch(*inputs):
    x, target = inputs
    return F.multilabel_margin_loss(x, target, reduction="none")


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
