from ntops_lab.kernels.fused.general._flash_attention import run_flash_attention


def run(*inputs):
    q, kv = inputs
    k = kv[:, :, :, 0, :]
    v = kv[:, :, :, 1, :]
    return run_flash_attention(q, k, v, is_causal=True, softmax_scale=1.0)
