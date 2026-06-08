import argparse
import importlib
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ntops_lab import list_operators  # noqa: E402

@dataclass
class Result:
    op: str
    category: str
    status: str
    elapsed_s: float
    detail: str = ""

def spec_module_name(record) -> str:
    spec_path = record.file.replace("kernels/", "specs/")[:-3].replace("/", ".")
    return "ntops_lab.testing." + spec_path

def select_records(args: argparse.Namespace):
    records = list_operators(status="done", category=args.category)

    if args.only:
        wanted = set(args.only)
        records = [record for record in records if record.op in wanted]

    if args.skip:
        skipped = set(args.skip)
        records = [record for record in records if record.op not in skipped]

    if args.start_at:
        for index, record in enumerate(records):
            if record.op == args.start_at:
                records = records[index:]
                break
        else:
            raise SystemExit(f"--start-at operator not found: {args.start_at}")

    if args.limit is not None:
        records = records[: args.limit]

    return records

def run_one(record, *, traceback_on_error: bool) -> Result:
    started = time.perf_counter()
    module_name = spec_module_name(record)

    try:
        module = importlib.import_module(module_name)
        check = getattr(module, "check", None)
        if check is None:
            return Result(record.op, record.category, "SKIP", time.perf_counter() - started, "no check()")
        check()
        return Result(record.op, record.category, "PASS", time.perf_counter() - started)
    except Exception as exc:  # noqa: BLE001
        detail = traceback.format_exc() if traceback_on_error else f"{type(exc).__name__}: {exc}"
        return Result(record.op, record.category, "FAIL", time.perf_counter() - started, detail)

def main() -> int:
    parser = argparse.ArgumentParser(description="Run correctness checks for all runnable ntops.lab operators.")
    parser.add_argument("--category", help="Only run one category, for example linear or pointwise")
    parser.add_argument("--limit", type=int, help="Run at most N selected operators")
    parser.add_argument("--start-at", help="Resume from this operator name in manifest order")
    parser.add_argument("--only", nargs="+", help="Run only these operator names")
    parser.add_argument("--skip", nargs="+", help="Skip these operator names")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed operator")
    parser.add_argument("--traceback", action="store_true", help="Print full traceback for failures")
    parser.add_argument("--list", action="store_true", help="List selected operators without running them")
    args = parser.parse_args()

    records = select_records(args)
    if args.list:
        for record in records:
            print(f"{record.op}\t{record.category}\t{record.file}")
        print(f"selected={len(records)}")
        return 0

    if not records:
        print("No operators selected.")
        return 0

    started = time.perf_counter()
    results: list[Result] = []
    print(f"Running correctness checks: selected={len(records)}")

    for index, record in enumerate(records, start=1):
        print(f"[{index}/{len(records)}] RUN {record.op} ({record.category})", flush=True)
        result = run_one(record, traceback_on_error=args.traceback)
        results.append(result)
        suffix = f" in {result.elapsed_s:.2f}s"
        if result.status == "PASS":
            print(f"[{index}/{len(records)}] PASS {record.op}{suffix}", flush=True)
        elif result.status == "SKIP":
            print(f"[{index}/{len(records)}] SKIP {record.op}{suffix}: {result.detail}", flush=True)
        else:
            print(f"[{index}/{len(records)}] FAIL {record.op}{suffix}: {result.detail}", flush=True)
            if args.fail_fast:
                break

    passed = sum(result.status == "PASS" for result in results)
    skipped = sum(result.status == "SKIP" for result in results)
    failed = [result for result in results if result.status == "FAIL"]
    elapsed = time.perf_counter() - started

    print()
    print("Correctness summary")
    print(f"  selected: {len(records)}")
    print(f"  executed: {len(results)}")
    print(f"  passed:   {passed}")
    print(f"  skipped:  {skipped}")
    print(f"  failed:   {len(failed)}")
    print(f"  elapsed:  {elapsed:.2f}s")

    if failed:
        print()
        print("Failed operators:")
        for result in failed:
            print(f"  - {result.op} ({result.category}): {result.detail.splitlines()[0]}")
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
