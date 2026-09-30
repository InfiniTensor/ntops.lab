from .var import run as run_var
from .mean import run as run_mean


def run(*inputs):
    """Return population variance and mean, using two NineToothed kernels."""
    return run_var(*inputs), run_mean(*inputs)
