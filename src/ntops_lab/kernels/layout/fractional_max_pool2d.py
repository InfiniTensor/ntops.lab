from ntops_lab.kernels.layout.avg_pool2d import _pool2d

def run(*inputs, kernel_size=2, output_size=None, output_ratio=None, return_indices=False, _random_samples=None):
    x = inputs[0]
    if len(inputs) > 1 and _random_samples is None:
        _random_samples = inputs[1]
    if kernel_size not in (2, (2, 2)):
        raise ValueError("fractional_max_pool2d currently supports kernel_size=2")
    if output_ratio is not None or return_indices:
        raise ValueError("fractional_max_pool2d currently supports output_size with return_indices=False")
    if x.shape[-2] % 2 != 0 or x.shape[-1] % 2 != 0:
        raise ValueError("fractional_max_pool2d currently requires even spatial dimensions")
    if output_size is not None and tuple(output_size) != (x.shape[-2] // 2, x.shape[-1] // 2):
        raise ValueError("fractional_max_pool2d currently supports output_size=input_size//2")
    return _pool2d(x, "max", kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
