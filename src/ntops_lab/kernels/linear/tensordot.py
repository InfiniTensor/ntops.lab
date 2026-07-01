from ntops_lab.kernels.linear.mm import run as _mm

def run(*inputs, dims=1):
    x, y = inputs
    if dims != 1:
        raise ValueError("tensordot currently supports dims=1")
    return _mm(x.reshape(-1, x.shape[-1]), y.reshape(y.shape[0], -1)).reshape(x.shape[0], y.shape[1])
