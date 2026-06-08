import ast
from pathlib import Path

from ntops_lab import categories, get_operator, list_operators, runnable_operators, todo_operators
from ntops_lab.registry import OPERATOR_MODULES

def test_manifest_counts_are_current() -> None:
    records = list_operators()
    assert len(records) == 246
    assert len(runnable_operators()) == 246
    assert len(todo_operators()) == 0

def test_categories_include_major_groups() -> None:
    counts = categories()
    assert counts["pointwise"] == 162
    assert counts["layout"] == 21
    assert counts["reduction"] == 24
    assert counts["fused/general"] == 9

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
    for path in [*root.glob("scripts/*.py"), *root.glob("tests/*.py"), *root.glob("src/**/*.py")]:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
