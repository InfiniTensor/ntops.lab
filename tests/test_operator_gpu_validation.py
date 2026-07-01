import importlib
import os

import pytest

from ntops_lab import runnable_operators


RUN_OPERATOR_VALIDATION = os.environ.get("NTOPS_RUN_OPERATOR_VALIDATION") == "1"


def spec_module_name(record) -> str:
    spec_path = record.file.replace("kernels/", "specs/")[:-3].replace("/", ".")
    return "ntops_lab.testing." + spec_path


@pytest.mark.parametrize("record", runnable_operators(), ids=lambda item: item.op)
def test_operator_matches_pytorch(record):
    if not RUN_OPERATOR_VALIDATION:
        pytest.skip("set NTOPS_RUN_OPERATOR_VALIDATION=1 to run GPU operator validation")
    module = importlib.import_module(spec_module_name(record))
    module.check()
