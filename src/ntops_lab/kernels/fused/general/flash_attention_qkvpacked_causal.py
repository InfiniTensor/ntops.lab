from ntops_lab.kernels.fused.general._flash_attention import run_flash_attention


def run(*inputs):
    (qkv,) = inputs
    q = qkv[:, :, :, 0, :]
    k = qkv[:, :, :, 1, :]
    v = qkv[:, :, :, 2, :]
    return run_flash_attention(q, k, v, is_causal=True, softmax_scale=1.0)
