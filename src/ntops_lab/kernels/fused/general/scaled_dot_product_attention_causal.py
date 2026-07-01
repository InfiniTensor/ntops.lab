from ntops_lab.kernels.fused.general.scaled_dot_product_attention import run as scaled_dot_product_attention


def run(*inputs):
    return scaled_dot_product_attention(*inputs, is_causal=True, scale=1.0)
