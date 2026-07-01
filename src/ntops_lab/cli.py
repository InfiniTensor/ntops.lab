import argparse
from collections import Counter

from .cache import cache_stats, format_size, ninetoothed_cache_path, triton_cache_path
from .catalog import categories, list_operators

def cmd_list(args: argparse.Namespace) -> int:
    records = list_operators(status=args.status, category=args.category)
    for item in records:
        print(f"{item.op}\t{item.status}\t{item.category}\t{item.file}")
    return 0

def cmd_summary(args: argparse.Namespace) -> int:
    records = list_operators()
    status_counts = Counter(item.status for item in records)
    print(f"total: {len(records)}")
    for status, count in sorted(status_counts.items()):
        print(f"{status}: {count}")
    print()
    for category, count in categories(records).items():
        runnable = sum(1 for item in records if item.category == category and item.runnable)
        print(f"{category}: {runnable}/{count} runnable")
    return 0

def cmd_cache_status(args: argparse.Namespace) -> int:
    caches = (
        ("NineToothed source", ninetoothed_cache_path()),
        ("Triton compiled", triton_cache_path()),
    )
    for label, path in caches:
        stats = cache_stats(path)
        print(f"{label}: {stats.files} files, {format_size(stats.bytes)}, {stats.path}")
    return 0

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ntops-lab")
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list", help="List operators")
    list_parser.add_argument("--status", choices=["done", "todo"])
    list_parser.add_argument("--category")
    list_parser.set_defaults(func=cmd_list)

    summary_parser = sub.add_parser("summary", help="Print coverage summary")
    summary_parser.set_defaults(func=cmd_summary)

    cache_parser = sub.add_parser("cache", help="Inspect compilation caches")
    cache_sub = cache_parser.add_subparsers(dest="cache_command", required=True)

    cache_status_parser = cache_sub.add_parser("status", help="Show cache paths and sizes")
    cache_status_parser.set_defaults(func=cmd_cache_status)
    return parser

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())
