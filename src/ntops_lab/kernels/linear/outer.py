from ntops_lab.kernels.linear.mm import run as _run_mm


def run(*inputs):
    x, y = inputs
    x_col = x.reshape((x.shape[0], 1)).contiguous()
    y_row = y.reshape((1, y.shape[0])).contiguous()
    return _run_mm(x_col, y_row)
