from ntops_lab.kernels.layout.upsample_nearest2d import run as _upsample_nearest2d

def run(*inputs, size=None, scale_factor=2, mode="nearest", align_corners=None, recompute_scale_factor=None, antialias=False):
    if size is not None:
        raise ValueError("interpolate currently supports scale_factor only")
    if mode != "nearest":
        raise ValueError("interpolate currently supports mode='nearest'")
    if align_corners is not None or recompute_scale_factor is not None or antialias:
        raise ValueError("unsupported interpolate option for this ntops_lab kernel")
    return _upsample_nearest2d(*inputs, scale_factor=scale_factor)
