from ntops_lab.kernels.layout.avg_pool2d import _pool2d

def run(*inputs):
    (x,) = inputs
    out2 = _pool2d(x.unsqueeze(2), "max", kernel_size=(1, 4), stride=(1, 2), padding=(0, 0))
    return out2.squeeze(2)
