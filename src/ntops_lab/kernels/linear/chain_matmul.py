from ntops_lab.kernels.linear.mm import run as _mm

def run(*inputs):
    a, b, c = inputs
    return _mm(_mm(a, b), c)
