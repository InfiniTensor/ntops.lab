def layernorm(*args, **kwargs):
    from ntops_lab.kernels.normalization.layernorm import run
    return run(*args, **kwargs)

def rms_norm(*args, **kwargs):
    from ntops_lab.kernels.normalization.rms_norm import run
    return run(*args, **kwargs)

__all__ = ['layernorm', 'rms_norm']
