from pathlib import Path

from ntops_lab import get_operator
from ntops_lab.registry import OPERATOR_MODULES

def test_manifest_points_to_kernel_modules() -> None:
    for module in OPERATOR_MODULES.values():
        assert module.startswith("ntops_lab.kernels.")

def test_api_package_exposes_known_operator() -> None:
    import ntops_lab.ops as ops

    assert callable(ops.get_op("add"))
    assert callable(ops.add)

def test_specs_are_separate_from_kernels() -> None:
    root = Path(__file__).resolve().parents[1]
    record = get_operator("add")
    kernel_path = root / "src" / "ntops_lab" / record.file
    spec_path = root / "src" / "ntops_lab" / "testing" / record.file.replace("kernels/", "specs/")
    assert kernel_path.exists()
    assert spec_path.exists()
    assert "def run_pytorch" not in kernel_path.read_text(encoding="utf-8")
    assert "def run_pytorch" in spec_path.read_text(encoding="utf-8")
