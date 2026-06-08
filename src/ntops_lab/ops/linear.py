def addbmm(*args, **kwargs):
    from ntops_lab.kernels.linear.addbmm import run
    return run(*args, **kwargs)

def addmm(*args, **kwargs):
    from ntops_lab.kernels.linear.addmm import run
    return run(*args, **kwargs)

def addmv(*args, **kwargs):
    from ntops_lab.kernels.linear.addmv import run
    return run(*args, **kwargs)

def addr(*args, **kwargs):
    from ntops_lab.kernels.linear.addr import run
    return run(*args, **kwargs)

def baddbmm(*args, **kwargs):
    from ntops_lab.kernels.linear.baddbmm import run
    return run(*args, **kwargs)

def bmm(*args, **kwargs):
    from ntops_lab.kernels.linear.bmm import run
    return run(*args, **kwargs)

def bmm_out(*args, **kwargs):
    from ntops_lab.kernels.linear.bmm_out import run
    return run(*args, **kwargs)

def dot(*args, **kwargs):
    from ntops_lab.kernels.linear.dot import run
    return run(*args, **kwargs)

def gemm_bias(*args, **kwargs):
    from ntops_lab.kernels.linear.gemm_bias import run
    return run(*args, **kwargs)

def gemm_bias_gelu(*args, **kwargs):
    from ntops_lab.kernels.linear.gemm_bias_gelu import run
    return run(*args, **kwargs)

def gemm_bias_relu(*args, **kwargs):
    from ntops_lab.kernels.linear.gemm_bias_relu import run
    return run(*args, **kwargs)

def grouped_mm(*args, **kwargs):
    from ntops_lab.kernels.linear.grouped_mm import run
    return run(*args, **kwargs)

def linear(*args, **kwargs):
    from ntops_lab.kernels.linear.linear import run
    return run(*args, **kwargs)

def mm(*args, **kwargs):
    from ntops_lab.kernels.linear.mm import run
    return run(*args, **kwargs)

def mm_out(*args, **kwargs):
    from ntops_lab.kernels.linear.mm_out import run
    return run(*args, **kwargs)

def mv(*args, **kwargs):
    from ntops_lab.kernels.linear.mv import run
    return run(*args, **kwargs)

def outer(*args, **kwargs):
    from ntops_lab.kernels.linear.outer import run
    return run(*args, **kwargs)

def vdot(*args, **kwargs):
    from ntops_lab.kernels.linear.vdot import run
    return run(*args, **kwargs)

__all__ = ['addbmm', 'addmm', 'addmv', 'addr', 'baddbmm', 'bmm', 'bmm_out', 'dot', 'gemm_bias', 'gemm_bias_gelu', 'gemm_bias_relu', 'grouped_mm', 'linear', 'mm', 'mm_out', 'mv', 'outer', 'vdot']
