import torch
import torch.nn.functional as F
from ntops_lab.kernels.creation.one_hot import run

OP_NAME = "one_hot"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"
NUM_CLASSES = (16, 23)


def make_inputs(num_classes, rows=8192, device="cuda"):
    return (torch.randint(0, num_classes, (rows,), device=device, dtype=torch.long),)


def run_pytorch(*inputs, num_classes):
    (index,) = inputs
    return F.one_hot(index, num_classes=num_classes)


def check(atol=1.0e-3, rtol=1.0e-3):
    for num_classes in NUM_CLASSES:
        inputs = make_inputs(num_classes)
        actual = run(*inputs, num_classes=num_classes)
        expected = run_pytorch(*inputs, num_classes=num_classes)
        torch.cuda.synchronize()
        passed = torch.equal(actual, expected)
        max_abs_error = 0.0 if passed else (actual.to(torch.float32) - expected.to(torch.float32)).abs().max().item()
        print(OP_NAME, "num_classes=", num_classes, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max_abs_error)
        if not passed:
            raise AssertionError(OP_NAME)
