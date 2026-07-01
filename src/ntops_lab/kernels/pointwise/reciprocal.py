from ._elementwise_helpers import make_unary_kernel, run_unary


def application(x, out):
    out = 1.0 / x


kernel = make_unary_kernel(application, "reciprocal")


def run(*inputs):
    return run_unary(kernel, *inputs)
