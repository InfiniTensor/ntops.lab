from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

@dataclass(frozen=True)
class OperatorRecord:
    op: str
    status: str
    category: str
    file: str
    reason: str = ""

    @property
    def runnable(self) -> bool:
        return self.status == "done"

def manifest_path() -> Path:
    return Path(__file__).with_name("operator_manifest.json")

def list_operators(
    *,
    status: str | None = None,
    category: str | None = None,
) -> list[OperatorRecord]:
    data = json.loads(manifest_path().read_text())
    records = [
        OperatorRecord(
            op=item["op"],
            status=item["status"],
            category=item["category"],
            file=item["file"],
            reason=item.get("reason", ""),
        )
        for item in data
    ]
    if status is not None:
        records = [item for item in records if item.status == status]
    if category is not None:
        records = [item for item in records if item.category == category]
    return records

def runnable_operators() -> list[OperatorRecord]:
    return list_operators(status="done")

def todo_operators() -> list[OperatorRecord]:
    return list_operators(status="todo")

def categories(records: Iterable[OperatorRecord] | None = None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in records or list_operators():
        counts[item.category] = counts.get(item.category, 0) + 1
    return dict(sorted(counts.items()))

def get_operator(op: str) -> OperatorRecord:
    for item in list_operators():
        if item.op == op:
            return item
    raise KeyError(op)
