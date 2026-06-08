def _upsample_nearest_exact1d(*args, **kwargs):
    from ntops_lab.kernels.layout.upsample_nearest_exact1d import run
    return run(*args, **kwargs)

def alias_copy(*args, **kwargs):
    from ntops_lab.kernels.layout.alias_copy import run
    return run(*args, **kwargs)

def avg_pool2d(*args, **kwargs):
    from ntops_lab.kernels.layout.avg_pool2d import run
    return run(*args, **kwargs)

def avg_pool3d(*args, **kwargs):
    from ntops_lab.kernels.layout.avg_pool3d import run
    return run(*args, **kwargs)

def contiguous(*args, **kwargs):
    from ntops_lab.kernels.layout.contiguous import run
    return run(*args, **kwargs)

def conv1d(*args, **kwargs):
    from ntops_lab.kernels.layout.conv1d import run
    return run(*args, **kwargs)

def conv2d(*args, **kwargs):
    from ntops_lab.kernels.layout.conv2d import run
    return run(*args, **kwargs)

def conv3d(*args, **kwargs):
    from ntops_lab.kernels.layout.conv3d import run
    return run(*args, **kwargs)

def conv_depthwise2d(*args, **kwargs):
    from ntops_lab.kernels.layout.conv_depthwise2d import run
    return run(*args, **kwargs)

def conv_transpose1d(*args, **kwargs):
    from ntops_lab.kernels.layout.conv_transpose1d import run
    return run(*args, **kwargs)

def conv_transpose2d(*args, **kwargs):
    from ntops_lab.kernels.layout.conv_transpose2d import run
    return run(*args, **kwargs)

def copy(*args, **kwargs):
    from ntops_lab.kernels.layout.copy import run
    return run(*args, **kwargs)

def cudnn_convolution(*args, **kwargs):
    from ntops_lab.kernels.layout.cudnn_convolution import run
    return run(*args, **kwargs)

def max_pool2d_with_indices(*args, **kwargs):
    from ntops_lab.kernels.layout.max_pool2d_with_indices import run
    return run(*args, **kwargs)

def max_pool3d_with_indices(*args, **kwargs):
    from ntops_lab.kernels.layout.max_pool3d_with_indices import run
    return run(*args, **kwargs)

def resolve_conj(*args, **kwargs):
    from ntops_lab.kernels.layout.resolve_conj import run
    return run(*args, **kwargs)

def to(*args, **kwargs):
    from ntops_lab.kernels.layout.to import run
    return run(*args, **kwargs)

def upsample_linear1d(*args, **kwargs):
    from ntops_lab.kernels.layout.upsample_linear1d import run
    return run(*args, **kwargs)

def upsample_nearest1d(*args, **kwargs):
    from ntops_lab.kernels.layout.upsample_nearest1d import run
    return run(*args, **kwargs)

def upsample_nearest2d(*args, **kwargs):
    from ntops_lab.kernels.layout.upsample_nearest2d import run
    return run(*args, **kwargs)

def upsample_nearest3d(*args, **kwargs):
    from ntops_lab.kernels.layout.upsample_nearest3d import run
    return run(*args, **kwargs)

__all__ = ['_upsample_nearest_exact1d', 'alias_copy', 'avg_pool2d', 'avg_pool3d', 'contiguous', 'conv1d', 'conv2d', 'conv3d', 'conv_depthwise2d', 'conv_transpose1d', 'conv_transpose2d', 'copy', 'cudnn_convolution', 'max_pool2d_with_indices', 'max_pool3d_with_indices', 'resolve_conj', 'to', 'upsample_linear1d', 'upsample_nearest1d', 'upsample_nearest2d', 'upsample_nearest3d']
