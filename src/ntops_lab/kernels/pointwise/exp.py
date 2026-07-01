import ninetoothed.language as ntl

from ._elementwise_helpers import make_unary_kernel, run_unary


def application(x, out):
    out = ntl.exp(x)


kernel = make_unary_kernel(application, "exp")


def run(*inputs):
    return run_unary(kernel, *inputs)
