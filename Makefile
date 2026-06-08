.PHONY: check test summary smoke-one smoke-pointwise

check:
	python scripts/check_manifest.py
	python -m compileall -q src scripts tests

test:
	python -m pytest

summary:
	python -m ntops_lab.cli summary

smoke-one:
	python scripts/run_operator.py add

smoke-pointwise:
	python scripts/run_smoke_suite.py --category pointwise --limit 10 --continue-on-error
