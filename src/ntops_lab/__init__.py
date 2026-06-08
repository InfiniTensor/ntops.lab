"""ntops.lab: experimental NineToothed operator implementations."""

from .catalog import (
    OperatorRecord,
    categories,
    get_operator,
    list_operators,
    manifest_path,
    runnable_operators,
    todo_operators,
)

def get_torch_op(name: str):
    from .ops import get_op

    return get_op(name)

def call_torch_op(name: str, *args, **kwargs):
    from .ops import call

    return call(name, *args, **kwargs)

__all__ = [
    "OperatorRecord",
    "call_torch_op",
    "categories",
    "get_operator",
    "get_torch_op",
    "list_operators",
    "manifest_path",
    "runnable_operators",
    "todo_operators",
]
