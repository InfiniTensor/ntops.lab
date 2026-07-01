from ntops_lab.kernels.layout.avg_pool2d import _pool2d

def run(*inputs):
    (x,) = inputs
    return _pool2d(x, "max", kernel_size=(4, 4), stride=(4, 4), padding=(0, 0))
