from ntops_lab.kernels.layout.conv1d import run as _conv1d

def run(*inputs, pad=1):
    x, weight, bias = inputs
    if pad != 1:
        raise ValueError("conv_tbc currently supports pad=1")
    y = _conv1d(x.permute(1, 2, 0), weight.permute(2, 1, 0), bias)
    return y.permute(2, 0, 1)
