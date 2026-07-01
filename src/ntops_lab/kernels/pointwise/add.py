from ._elementwise_helpers import make_binary_kernel, run_binary


def application(x, y, out):
    out = x + y


kernel = make_binary_kernel(application, "add")


def run(*inputs):
    return run_binary(kernel, *inputs)
