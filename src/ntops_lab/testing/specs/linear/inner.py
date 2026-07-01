import torch
from ntops_lab.kernels.linear.inner import run

OP_NAME = "inner"
IMPLEMENTATION_STATUS = "implemented_torch_top_batch1"

def make_inputs(device="cuda", dtype=torch.float32):
    return torch.randn((128, 64), device=device, dtype=torch.float16), torch.randn((256, 64), device=device, dtype=torch.float16)

def run_pytorch(*inputs):
    device = inputs[0].device if inputs and isinstance(inputs[0], torch.Tensor) else "cuda"
    dtype = inputs[0].dtype if inputs and isinstance(inputs[0], torch.Tensor) and torch.is_floating_point(inputs[0]) else torch.float32
    return torch.inner(inputs[0], inputs[1])

def check(atol=1.0e-1, rtol=1.0e-1):
    inputs = make_inputs()
    actual = run(*inputs)
    expected = run_pytorch(*inputs)
    torch.cuda.synchronize()
    actual_items = actual if isinstance(actual, tuple) else (actual,)
    expected_items = expected if isinstance(expected, tuple) else (expected,)
    passed = len(actual_items) == len(expected_items)
    max_abs_error = 0.0
    for actual_item, expected_item in zip(actual_items, expected_items):
        if torch.is_floating_point(expected_item) or torch.is_complex(expected_item):
            diff = (actual_item - expected_item).abs()
            max_abs_error = max(max_abs_error, diff.max().item())
            passed = passed and torch.allclose(actual_item, expected_item, atol=atol, rtol=rtol)
        else:
            passed = passed and torch.equal(actual_item, expected_item)
    print(OP_NAME, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max_abs_error)
    if not passed:
        raise AssertionError(OP_NAME)
