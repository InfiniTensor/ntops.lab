import torch
from ntops_lab.kernels.fused.fla.cumsum import run, BT

H = 16
T = 1024
B = 4
OP_NAME = "cumsum"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(device="cuda", dtype=torch.float32):
    return (torch.randn((B, H, T), device=device, dtype=dtype),)

def run_pytorch(*inputs):
    (x,) = inputs
    chunks = []
    for start in range(0, x.shape[-1], BT):
        chunks.append(torch.cumsum(x[..., start:start + BT], dim=-1))
    return torch.cat(chunks, dim=-1)

def check(atol=1.0e-5, rtol=1.0e-5):
    inputs = make_inputs()
    actual = run(*inputs)
    expected = run_pytorch(*inputs)
    torch.cuda.synchronize()
    max_abs_error = (actual - expected).abs().max().item()
    passed = torch.allclose(actual, expected, atol=atol, rtol=rtol)
    print(OP_NAME, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max_abs_error)
    if not passed:
        raise AssertionError(OP_NAME)
