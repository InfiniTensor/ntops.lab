from ntops_lab.kernels.linear.addmm import run as _run_addmm


def run(*inputs, beta=1.0, alpha=1.0):
    c, x, y = inputs
    if beta != 1.0 or alpha != 1.0:
        raise NotImplementedError("addr currently supports alpha=1.0 and beta=1.0")
    x_col = x.reshape((x.shape[0], 1)).contiguous()
    y_row = y.reshape((1, y.shape[0])).contiguous()
    return _run_addmm(c, x_col, y_row)
