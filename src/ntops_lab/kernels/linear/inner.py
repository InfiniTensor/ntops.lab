from ntops_lab.kernels.linear.mm import run as _mm

def run(*inputs):
    x, y = inputs
    return _mm(x, y.transpose(0, 1))
