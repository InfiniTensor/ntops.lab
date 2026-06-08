def normalize_scale_factor(scale_factor, dims):
    values = scale_factor if isinstance(scale_factor, tuple) else (scale_factor,) * dims
    if len(values) != dims:
        raise ValueError(f"scale_factor must have {dims} value(s)")
    normalized = []
    for value in values:
        if isinstance(value, bool) or int(value) != value:
            raise ValueError("ntops.lab upsample kernels currently require integer scale factors")
        value = int(value)
        if value <= 0:
            raise ValueError("scale_factor must be positive")
        normalized.append(value)
    return tuple(normalized)
