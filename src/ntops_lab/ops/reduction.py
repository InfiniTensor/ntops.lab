def _safe_softmax(*args, **kwargs):
    from ntops_lab.kernels.reduction.safe_softmax import run
    return run(*args, **kwargs)

def all_dim(*args, **kwargs):
    from ntops_lab.kernels.reduction.all_dim import run
    return run(*args, **kwargs)

def all_dims(*args, **kwargs):
    from ntops_lab.kernels.reduction.all_dims import run
    return run(*args, **kwargs)

def amax(*args, **kwargs):
    from ntops_lab.kernels.reduction.amax import run
    return run(*args, **kwargs)

def aminmax(*args, **kwargs):
    from ntops_lab.kernels.reduction.aminmax import run
    return run(*args, **kwargs)

def any(*args, **kwargs):
    from ntops_lab.kernels.reduction.any import run
    return run(*args, **kwargs)

def any_dim(*args, **kwargs):
    from ntops_lab.kernels.reduction.any_dim import run
    return run(*args, **kwargs)

def any_dims(*args, **kwargs):
    from ntops_lab.kernels.reduction.any_dims import run
    return run(*args, **kwargs)

def log_softmax(*args, **kwargs):
    from ntops_lab.kernels.reduction.log_softmax import run
    return run(*args, **kwargs)

def logsumexp(*args, **kwargs):
    from ntops_lab.kernels.reduction.logsumexp import run
    return run(*args, **kwargs)

def max(*args, **kwargs):
    from ntops_lab.kernels.reduction.max import run
    return run(*args, **kwargs)

def max_dim(*args, **kwargs):
    from ntops_lab.kernels.reduction.max_dim import run
    return run(*args, **kwargs)

def mean(*args, **kwargs):
    from ntops_lab.kernels.reduction.mean import run
    return run(*args, **kwargs)

def mean_dim(*args, **kwargs):
    from ntops_lab.kernels.reduction.mean_dim import run
    return run(*args, **kwargs)

def mean_dim_comm(*args, **kwargs):
    from ntops_lab.kernels.reduction.mean_dim_comm import run
    return run(*args, **kwargs)

def min(*args, **kwargs):
    from ntops_lab.kernels.reduction.min import run
    return run(*args, **kwargs)

def min_dim(*args, **kwargs):
    from ntops_lab.kernels.reduction.min_dim import run
    return run(*args, **kwargs)

def scaled_softmax(*args, **kwargs):
    from ntops_lab.kernels.reduction.scaled_softmax import run
    return run(*args, **kwargs)

def softmax(*args, **kwargs):
    from ntops_lab.kernels.reduction.softmax import run
    return run(*args, **kwargs)

def std(*args, **kwargs):
    from ntops_lab.kernels.reduction.std import run
    return run(*args, **kwargs)

def sum(*args, **kwargs):
    from ntops_lab.kernels.reduction.sum import run
    return run(*args, **kwargs)

def var(*args, **kwargs):
    from ntops_lab.kernels.reduction.var import run
    return run(*args, **kwargs)

def var_mean(*args, **kwargs):
    from ntops_lab.kernels.reduction.var_mean import run
    return run(*args, **kwargs)

def vector_norm(*args, **kwargs):
    from ntops_lab.kernels.reduction.vector_norm import run
    return run(*args, **kwargs)

__all__ = ['_safe_softmax', 'all_dim', 'all_dims', 'amax', 'aminmax', 'any', 'any_dim', 'any_dims', 'log_softmax', 'logsumexp', 'max', 'max_dim', 'mean', 'mean_dim', 'mean_dim_comm', 'min', 'min_dim', 'scaled_softmax', 'softmax', 'std', 'sum', 'var', 'var_mean', 'vector_norm']
