from ntops_lab.kernels.layout.upsample_nearest2d import run as _upsample_nearest2d

def run(*inputs, size=None, scale_factor=2):
    if size is not None:
        raise ValueError("upsample_nearest currently supports scale_factor only")
    return _upsample_nearest2d(*inputs, scale_factor=scale_factor)
