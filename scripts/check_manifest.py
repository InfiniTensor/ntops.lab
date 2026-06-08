import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "ntops_lab"
MANIFEST = PACKAGE / "operator_manifest.json"

def main() -> int:
    data = json.loads(MANIFEST.read_text())
    errors: list[str] = []
    seen: set[str] = set()

    for item in data:
        op = item["op"]
        if op in seen:
            errors.append(f"duplicate op: {op}")
        seen.add(op)

        rel = item["file"]
        path = PACKAGE / rel
        if not path.exists():
            errors.append(f"missing file for {op}: {rel}")
        if item["status"] not in {"done", "todo"}:
            errors.append(f"bad status for {op}: {item['status']}")

    print(f"operators: {len(data)}")
    print(dict(Counter(item["status"] for item in data)))
    print(dict(Counter(item["category"] for item in data)))

    if errors:
        for error in errors:
            print("ERROR:", error)
        return 1
    print("manifest ok")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
