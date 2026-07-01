import torch
import torch.nn.functional as F
from ntops_lab.kernels.fused.general.multi_head_attention_forward import run

OP_NAME = "multi_head_attention_forward"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"

def make_inputs(device="cuda", dtype=torch.float16):
    length = 128
    batch = 1
    embed = 64
    q = torch.randn((length, batch, embed), device=device, dtype=dtype) * 0.25
    k = torch.randn((length, batch, embed), device=device, dtype=dtype) * 0.25
    v = torch.randn((length, batch, embed), device=device, dtype=dtype) * 0.25
    in_w = torch.randn((3 * embed, embed), device=device, dtype=dtype) * 0.25
    in_b = torch.randn((3 * embed,), device=device, dtype=dtype) * 0.25
    out_w = torch.randn((embed, embed), device=device, dtype=dtype) * 0.25
    out_b = torch.randn((embed,), device=device, dtype=dtype) * 0.25
    return q, k, v, embed, 2, in_w, in_b, None, None, False, 0.0, out_w, out_b

def run_pytorch(*inputs):
    return F.multi_head_attention_forward(
        *inputs,
        training=False,
        key_padding_mask=None,
        need_weights=False,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
    )

def check(atol=2.0e-1, rtol=2.0e-1):
    inputs = make_inputs()
    actual, _ = run(*inputs, training=False, need_weights=False)
    expected, _ = run_pytorch(*inputs)
    torch.cuda.synchronize()
    max_abs_error = (actual - expected).abs().max().item()
    passed = torch.allclose(actual, expected, atol=atol, rtol=rtol)
    print(OP_NAME, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max_abs_error)
    if not passed:
        raise AssertionError(OP_NAME)
