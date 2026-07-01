import ninetoothed.language as ntl

from ._elementwise_helpers import make_unary_kernel, run_unary


def application(x, out):
    out = 2.0 / (1.0 + ntl.exp(-2.0 * x)) - 1.0


kernel = make_unary_kernel(application, "tanh")


def run(*inputs):
    return run_unary(kernel, *inputs)
