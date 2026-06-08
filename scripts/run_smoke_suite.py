import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ntops_lab import list_operators  # noqa: E402

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", help="Only run one category, such as pointwise")
    parser.add_argument("--limit", type=int, help="Run at most N operators")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    records = list_operators(status="done", category=args.category)
    if args.limit is not None:
        records = records[: args.limit]

    passed = failed = skipped = 0
    for record in records:
        module_name = "ntops_lab.testing." + record.file.replace("kernels/", "specs/")[:-3].replace("/", ".")
        try:
            module = importlib.import_module(module_name)
            check = getattr(module, "check", None)
            if check is None:
                skipped += 1
                print(record.op, "SKIP no check()")
                continue
            check()
            passed += 1
        except Exception as exc:
            failed += 1
            print(record.op, "FAILED", type(exc).__name__, exc)
            if not args.continue_on_error:
                break

    print("SUMMARY", "passed=", passed, "skipped=", skipped, "failed=", failed)
    return 1 if failed else 0

if __name__ == "__main__":
    raise SystemExit(main())
