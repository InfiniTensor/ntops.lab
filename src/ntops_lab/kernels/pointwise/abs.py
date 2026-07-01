import ninetoothed.language as ntl

from ._elementwise_helpers import make_unary_kernel, run_unary


def application(x, out):
    out = ntl.where(x < 0.0, -x, x)


kernel = make_unary_kernel(application, "abs")


def run(*inputs):
    return run_unary(kernel, *inputs)
