from ntops_lab.kernels.fused.general._flash_attention import run_flash_attention


def run(*inputs):
    q, k, v = inputs
    return run_flash_attention(q, k, v, is_causal=False, softmax_scale=1.0)
