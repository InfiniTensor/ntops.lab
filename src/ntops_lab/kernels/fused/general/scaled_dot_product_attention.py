from ntops_lab.kernels.fused.general._flash_attention import run_flash_attention


def run(*inputs, attn_mask=None, dropout_p=0.0, is_causal=False, scale=1.0, enable_gqa=False):
    q, k, v = inputs
    if attn_mask is not None or dropout_p != 0.0 or enable_gqa:
        raise ValueError("scaled_dot_product_attention currently supports attn_mask=None, dropout_p=0, enable_gqa=False")
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return run_flash_attention(q, k, v, is_causal=is_causal, softmax_scale=scale)
