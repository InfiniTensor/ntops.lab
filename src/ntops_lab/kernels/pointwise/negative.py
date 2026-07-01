from ._elementwise_helpers import make_unary_kernel, run_unary


def application(x, out):
    out = -x


kernel = make_unary_kernel(application, "negative")


def run(*inputs):
    return run_unary(kernel, *inputs)
