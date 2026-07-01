from ntops_lab.kernels.fused.general.scaled_dot_product_attention import run as sdpa
from ntops_lab.kernels.linear.linear import run as linear

def _project(x, weight, bias, num_heads):
    length, batch, embed = x.shape
    flat = x.reshape(length * batch, embed)
    proj = linear(flat, weight, bias)
    head_dim = embed // num_heads
    return proj.reshape(length, batch, num_heads, head_dim).permute(1, 2, 0, 3).contiguous()

def run(
    query,
    key,
    value,
    embed_dim_to_check,
    num_heads,
    in_proj_weight,
    in_proj_bias,
    bias_k,
    bias_v,
    add_zero_attn,
    dropout_p,
    out_proj_weight,
    out_proj_bias,
    training=False,
    key_padding_mask=None,
    need_weights=False,
    attn_mask=None,
    use_separate_proj_weight=False,
    q_proj_weight=None,
    k_proj_weight=None,
    v_proj_weight=None,
    static_k=None,
    static_v=None,
    average_attn_weights=True,
    is_causal=False,
):
    if (
        bias_k is not None
        or bias_v is not None
        or add_zero_attn
        or dropout_p != 0.0
        or key_padding_mask is not None
        or attn_mask is not None
        or use_separate_proj_weight
        or static_k is not None
        or static_v is not None
        or need_weights
    ):
        raise ValueError("multi_head_attention_forward currently supports packed qkv, no masks/dropout, need_weights=False")
    if query.shape[-1] != embed_dim_to_check:
        raise ValueError("query embedding dim does not match embed_dim_to_check")
    embed = query.shape[-1]
    head_dim = embed // num_heads
    if embed != 64 or head_dim != 32:
        raise ValueError("multi_head_attention_forward currently supports embed_dim=64, num_heads=2")

    q_weight = in_proj_weight[:embed, :]
    k_weight = in_proj_weight[embed : 2 * embed, :]
    v_weight = in_proj_weight[2 * embed :, :]
    q_bias = in_proj_bias[:embed] if in_proj_bias is not None else None
    k_bias = in_proj_bias[embed : 2 * embed] if in_proj_bias is not None else None
    v_bias = in_proj_bias[2 * embed :] if in_proj_bias is not None else None

    q = _project(query, q_weight, q_bias, num_heads)
    k = _project(key, k_weight, k_bias, num_heads)
    v = _project(value, v_weight, v_bias, num_heads)
    attn = sdpa(q * (head_dim ** -0.5), k, v, is_causal=is_causal, scale=1.0)
    length, batch = query.shape[0], query.shape[1]
    merged = attn.permute(2, 0, 1, 3).reshape(length * batch, embed)
    out = linear(merged, out_proj_weight, out_proj_bias).reshape(length, batch, embed)
    return out, None
