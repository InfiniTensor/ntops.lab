from ntops_lab.kernels.layout.avg_pool3d import _pool3d


def run(*inputs):
    (x,) = inputs
    return _pool3d(x, "max", kernel_size=(2, 4, 4), stride=(2, 4, 4), padding=(0, 0, 0))
