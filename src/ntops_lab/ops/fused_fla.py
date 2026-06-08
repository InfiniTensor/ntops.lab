def cumsum(*args, **kwargs):
    from ntops_lab.kernels.fused.fla.cumsum import run
    return run(*args, **kwargs)

__all__ = ['cumsum']
