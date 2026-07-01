import ast
from pathlib import Path

from ntops_lab import categories, get_operator, list_operators, runnable_operators, todo_operators
from ntops_lab.registry import OPERATOR_MODULES


EXPECTED_COUNTS = {'creation': 11, 'fused/fla': 1, 'fused/general': 283, 'layout': 91, 'linear': 98, 'normalization': 6, 'pointwise': 457, 'reduction': 177}


def test_manifest_counts_are_current() -> None:
    records = list_operators()
    assert len(records) == 1124
    assert len(runnable_operators()) == 1124
    assert len(todo_operators()) == 0


def test_categories_are_current() -> None:
    assert categories() == EXPECTED_COUNTS


def test_operator_files_exist() -> None:
    package_root = Path(__file__).resolve().parents[1] / "src" / "ntops_lab"
    for item in list_operators():
        assert (package_root / item.file).exists(), item


def test_lookup_known_operator() -> None:
    add = get_operator("add")
    assert add.runnable
    assert add.category == "pointwise"


def test_registry_matches_manifest() -> None:
    manifest_ops = {item.op for item in list_operators()}
    assert set(OPERATOR_MODULES) == manifest_ops
    assert all(module.startswith("ntops_lab.kernels.") for module in OPERATOR_MODULES.values())


def test_all_committed_python_sources_parse() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in [*root.glob("tests/*.py"), *root.glob("src/**/*.py")]:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_legacy_script_test_runners_are_removed() -> None:
    root = Path(__file__).resolve().parents[1]
    legacy_scripts = {
        "check_manifest.py",
        "run_all_correctness.py",
        "run_operator.py",
        "run_smoke_suite.py",
    }
    assert not any((root / "scripts" / name).exists() for name in legacy_scripts)
