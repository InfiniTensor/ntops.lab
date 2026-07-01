# Auto-generated from operator_manifest.json. Do not edit by hand.

def batch_norm(*args, **kwargs):
    from ntops_lab.kernels.normalization.batch_norm import run
    return run(*args, **kwargs)

def group_norm(*args, **kwargs):
    from ntops_lab.kernels.normalization.group_norm import run
    return run(*args, **kwargs)

def instance_norm(*args, **kwargs):
    from ntops_lab.kernels.normalization.instance_norm import run
    return run(*args, **kwargs)

def layer_norm(*args, **kwargs):
    from ntops_lab.kernels.normalization.layer_norm import run
    return run(*args, **kwargs)

def local_response_norm(*args, **kwargs):
    from ntops_lab.kernels.normalization.local_response_norm import run
    return run(*args, **kwargs)

def rms_norm(*args, **kwargs):
    from ntops_lab.kernels.normalization.rms_norm import run
    return run(*args, **kwargs)
