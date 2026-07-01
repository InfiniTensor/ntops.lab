from ntops_lab.kernels.layout.upsample_nearest2d import run as _upsample_nearest2d

def run(*inputs, size=None, scale_factor=2, mode="nearest", align_corners=None):
    if size is not None:
        raise ValueError("upsample currently supports scale_factor only")
    if mode != "nearest":
        raise ValueError("upsample currently supports mode='nearest'")
    if align_corners is not None:
        raise ValueError("unsupported upsample align_corners option for nearest mode")
    return _upsample_nearest2d(*inputs, scale_factor=scale_factor)
