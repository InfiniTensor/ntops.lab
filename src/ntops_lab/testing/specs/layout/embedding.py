import torch
import torch.nn.functional as F
from ntops_lab.kernels.layout.embedding import run

OP_NAME = "embedding"
IMPLEMENTATION_STATUS = "implemented_ninetoothed"
TEST_SHAPES = ((64, 16), (96, 24))


def make_inputs(vocab=64, dim=16, rows=8192, device="cuda", dtype=torch.float32):
    index = torch.randint(0, vocab, (rows,), device=device, dtype=torch.long)
    weight = torch.randn((vocab, dim), device=device, dtype=dtype) * 0.5
    return index, weight


def run_pytorch(*inputs):
    index, weight = inputs
    return F.embedding(index, weight)


def check(atol=1.0e-3, rtol=1.0e-3):
    for vocab, dim in TEST_SHAPES:
        inputs = make_inputs(vocab=vocab, dim=dim)
        actual = run(*inputs)
        expected = run_pytorch(*inputs)
        torch.cuda.synchronize()
        max_abs_error = (actual - expected).abs().max().item()
        passed = torch.allclose(actual, expected, atol=atol, rtol=rtol)
        print(OP_NAME, "vocab=", vocab, "dim=", dim, "status=", IMPLEMENTATION_STATUS, "passed=", passed, "max_abs_error=", max_abs_error)
        if not passed:
            raise AssertionError(OP_NAME)
