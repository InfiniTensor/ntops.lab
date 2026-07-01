import torch
from ntops_lab.kernels.reduction.mean_dim_comm import run

ROWS = 65536
OP_NAME = "mean_dim_comm"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"


TEST_DIMS = (32, 64)

def make_inputs(dim=32, device="cuda", dtype=torch.float32):
    if OP_NAME.startswith(("all", "any")):
        x = torch.randint(0, 2, (ROWS, dim), device=device, dtype=torch.int32).to(dtype)
    else:
        x = torch.randn((ROWS, dim), device=device, dtype=dtype)
    return (x,)

def run_pytorch(*inputs):
    (x,) = inputs
    return x.mean(dim=1)

def check(atol=1.0e-4, rtol=1.0e-4):
    for dim in TEST_DIMS:
        inputs = make_inputs(dim=dim)
        actual = run(*inputs)
        expected = run_pytorch(*inputs)
        torch.cuda.synchronize()
        if actual.dtype == torch.bool or expected.dtype == torch.bool:
            passed = torch.equal(actual.bool(), expected.bool())
            max_abs_error = 0.0
        else:
            max_abs_error = (actual - expected).abs().max().item()
            passed = torch.allclose(actual, expected, atol=atol, rtol=rtol)
        print(OP_NAME, "dim=", dim, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max_abs_error)
        if not passed:
            raise AssertionError(OP_NAME)
