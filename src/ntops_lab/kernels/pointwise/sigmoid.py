import ninetoothed.language as ntl

from ._elementwise_helpers import make_unary_kernel, run_unary


def application(x, out):
    out = 1.0 / (1.0 + ntl.exp(-x))


kernel = make_unary_kernel(application, "sigmoid")


def run(*inputs):
    return run_unary(kernel, *inputs)
