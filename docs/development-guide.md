# Development Guide

## Add A New Operator

1. Choose a category under `src/ntops_lab/kernels`.
2. Create one kernel file per operator.
3. Implement the kernel entry point:

```python
def run(*inputs): ...
```

4. Add a matching PyTorch-facing wrapper in `src/ntops_lab/ops`.
5. Add a matching test spec in `src/ntops_lab/testing/specs` with:

```python
def make_inputs(...): ...
def run_pytorch(*inputs): ...
def check(...): ...
```

6. Add the operator to `src/ntops_lab/operator_manifest.json`.
7. Run:

```bash
python -m pytest
NTOPS_RUN_OPERATOR_VALIDATION=1 python -m pytest tests/test_operator_gpu_validation.py -k my_op
```

## Unsupported Operators

Unsupported operators should not be committed as runnable kernel modules. Keep
their blocker analysis in documentation until there is a real implementation.

## Naming

- Use stable operator names in the manifest, such as `add_inplace` or
  `softmax_out`.
- Use filesystem-safe names for kernel and spec modules, such as
  `add_inplace.py`.
- Keep category names stable; coverage tooling depends on them.

## Validation Levels

- **Catalog validation**: no GPU required; run by default pytest.
- **Single-op GPU validation**: use `NTOPS_RUN_OPERATOR_VALIDATION=1` with a pytest `-k` selector.
- **Full GPU validation**: run `tests/test_operator_gpu_validation.py` with `NTOPS_RUN_OPERATOR_VALIDATION=1`.

Full GPU coverage is intentionally not part of default `pytest`, because kernel
compilation requires a CUDA/NineToothed environment.
