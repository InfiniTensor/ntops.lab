import torch
import torch.nn.functional as F

from ntops_lab.kernels.fused.general.flash_attention import run


OP_NAME = "flash_attention"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"


def make_inputs(device="cuda", dtype=torch.float16):
    q = torch.randn((1, 2, 128, 64), device=device, dtype=dtype) * 0.25
    k = torch.randn((1, 2, 128, 64), device=device, dtype=dtype) * 0.25
    v = torch.randn((1, 2, 128, 64), device=device, dtype=dtype) * 0.25
    return q, k, v


def run_actual(*inputs):
    q, k, v = inputs
    return run(q, k, v)


def run_pytorch(*inputs):
    q, k, v = inputs
    return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=1.0)


def check(atol=2.5e-2, rtol=2.5e-2):
    inputs = make_inputs()
    actual = run_actual(*inputs)
    expected = run_pytorch(*inputs)
    torch.cuda.synchronize()
    max_abs_error = (actual - expected).abs().max().item()
    passed = torch.allclose(actual, expected, atol=atol, rtol=rtol)
    print(OP_NAME, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max_abs_error)
    if not passed:
        raise AssertionError(OP_NAME)
