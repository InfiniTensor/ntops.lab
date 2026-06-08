def eye(*args, **kwargs):
    from ntops_lab.kernels.creation.eye import run
    return run(*args, **kwargs)

def eye_m(*args, **kwargs):
    from ntops_lab.kernels.creation.eye_m import run
    return run(*args, **kwargs)

def full(*args, **kwargs):
    from ntops_lab.kernels.creation.full import run
    return run(*args, **kwargs)

def full_like(*args, **kwargs):
    from ntops_lab.kernels.creation.full_like import run
    return run(*args, **kwargs)

def new_full(*args, **kwargs):
    from ntops_lab.kernels.creation.new_full import run
    return run(*args, **kwargs)

def ones(*args, **kwargs):
    from ntops_lab.kernels.creation.ones import run
    return run(*args, **kwargs)

def ones_like(*args, **kwargs):
    from ntops_lab.kernels.creation.ones_like import run
    return run(*args, **kwargs)

def zeros(*args, **kwargs):
    from ntops_lab.kernels.creation.zeros import run
    return run(*args, **kwargs)

def zeros_like(*args, **kwargs):
    from ntops_lab.kernels.creation.zeros_like import run
    return run(*args, **kwargs)

__all__ = ['eye', 'eye_m', 'full', 'full_like', 'new_full', 'ones', 'ones_like', 'zeros', 'zeros_like']
