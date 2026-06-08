def add_rms_norm(*args, **kwargs):
    from ntops_lab.kernels.fused.general.add_rms_norm import run
    return run(*args, **kwargs)

def fused_add_rms_norm(*args, **kwargs):
    from ntops_lab.kernels.fused.general.fused_add_rms_norm import run
    return run(*args, **kwargs)

def geglu(*args, **kwargs):
    from ntops_lab.kernels.fused.general.geglu import run
    return run(*args, **kwargs)

def gelu_and_mul(*args, **kwargs):
    from ntops_lab.kernels.fused.general.gelu_and_mul import run
    return run(*args, **kwargs)

def reglu(*args, **kwargs):
    from ntops_lab.kernels.fused.general.reglu import run
    return run(*args, **kwargs)

def rotary_embedding(*args, **kwargs):
    from ntops_lab.kernels.fused.general.rotary_embedding import run
    return run(*args, **kwargs)

def silu_and_mul(*args, **kwargs):
    from ntops_lab.kernels.fused.general.silu_and_mul import run
    return run(*args, **kwargs)

def skip_layernorm(*args, **kwargs):
    from ntops_lab.kernels.fused.general.skip_layernorm import run
    return run(*args, **kwargs)

def swiglu(*args, **kwargs):
    from ntops_lab.kernels.fused.general.swiglu import run
    return run(*args, **kwargs)

__all__ = ['add_rms_norm', 'fused_add_rms_norm', 'geglu', 'gelu_and_mul', 'reglu', 'rotary_embedding', 'silu_and_mul', 'skip_layernorm', 'swiglu']
