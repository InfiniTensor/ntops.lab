from .min import run as run_min
from .max import run as run_max


def run(*inputs):
    """Compute both row reductions with independent, supported output domains."""
    return run_min(*inputs), run_max(*inputs)
