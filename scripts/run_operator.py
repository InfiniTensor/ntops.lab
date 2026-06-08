import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ntops_lab import get_operator  # noqa: E402

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("op", help="Operator name from the manifest")
    args = parser.parse_args()

    record = get_operator(args.op)
    if not record.runnable:
        print(f"{record.op} is not runnable: {record.reason}")
        return 2

    module_name = "ntops_lab.testing." + record.file.replace("kernels/", "specs/")[:-3].replace("/", ".")
    module = importlib.import_module(module_name)
    if not hasattr(module, "check"):
        print(f"{record.op} has no check() function")
        return 3
    module.check()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
