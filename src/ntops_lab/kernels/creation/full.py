import ninetoothed
from ninetoothed import Tensor, block_size

BLOCK_SIZE = block_size()

def arrangement(out):
    return out.tile((BLOCK_SIZE,))

def application(out):
    out = 3.0

kernel = ninetoothed.make(arrangement, application, (Tensor(1),), kernel_name="ntops_lab_full")

def run(*inputs):
    out, = inputs
    kernel(out)
    return out
