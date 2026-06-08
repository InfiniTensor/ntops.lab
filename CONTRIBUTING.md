# Contributing

`ntops.lab` welcomes focused operator improvements.

Before opening a pull request:

1. Keep each operator in a separate file.
2. Add or update the PyTorch reference in the same module.
3. Update `src/ntops_lab/operator_manifest.json`.
4. Run:

```bash
python scripts/check_manifest.py
pytest
```

If you add a runnable GPU kernel, also run:

```bash
ntops-lab check <operator-name>
```

For blocked operators, leave a precise `REASON` explaining whether the blocker
is a missing NineToothed interface, a missing math primitive, or just manual
lowering work.
