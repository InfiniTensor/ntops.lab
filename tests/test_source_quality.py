import ast
import builtins
import importlib
import json
import re
import symtable
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PYTHON_ROOTS = (ROOT / "scripts", SRC, ROOT / "tests")


def python_files():
    for root in PYTHON_ROOTS:
        yield from root.rglob("*.py")


def symbol_tables(table):
    yield table
    for child in table.get_children():
        yield from symbol_tables(child)


def undefined_globals(path, text):
    top = symtable.symtable(text, str(path), "exec")
    defined = {
        symbol.get_name()
        for symbol in top.get_symbols()
        if symbol.is_assigned() or symbol.is_imported() or symbol.is_parameter()
    }
    allowed = defined | set(dir(builtins)) | {"__file__"}
    return {
        symbol.get_name()
        for table in symbol_tables(top)
        for symbol in table.get_symbols()
        if symbol.is_referenced() and symbol.is_global() and symbol.get_name() not in allowed
    }


def unused_imports(path, text):
    if path.name == "__init__.py":
        return set()
    tree = ast.parse(text)
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    unused = set()
    lines = text.splitlines()
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if "noqa" in lines[node.lineno - 1]:
            continue
        for alias in node.names:
            name = alias.asname or alias.name.split(".")[0]
            if name not in used and not (isinstance(node, ast.ImportFrom) and alias.name == "*"):
                unused.add(name)
    return unused


def unused_kernel_assignments(text):
    tree = ast.parse(text)
    loaded = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    unused = set()
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id not in loaded:
                unused.add(target.id)
    return unused


def test_python_source_quality():
    failures = []
    for path in python_files():
        relative = path.relative_to(ROOT)
        text = path.read_text(encoding="utf-8")
        if text.startswith(("\n", "\r\n")):
            failures.append(f"{relative}: leading blank line")
        if "from __future__ import " + "annotations" in text:
            failures.append(f"{relative}: unnecessary future annotations import")
        if "sys.path." + "pop(0)" in text:
            failures.append(f"{relative}: mutates sys.path")
        if "\n\n\n\n" in text:
            failures.append(f"{relative}: more than two consecutive newlines")
        if path.is_relative_to(SRC / "ntops_lab" / "kernels"):
            if re.search(r"^BLOCK_SIZE\s*=\s*\d+\s*$", text, re.MULTILINE):
                failures.append(f"{relative}: fixed generic BLOCK_SIZE")
            if "block_size()" in text and "max_num_configs=1" in text:
                failures.append(f"{relative}: dynamic block_size disabled by max_num_configs=1")
            if re.search(r"^SCALE\s*=", text, re.MULTILINE):
                failures.append(f"{relative}: fixed SCALE switch")
            if re.search(r"^OP_NAME\s*=", text, re.MULTILINE):
                failures.append(f"{relative}: fixed OP_NAME switch")
            if "if OP_NAME" in text or "OP_NAME.startswith" in text:
                failures.append(f"{relative}: OP_NAME-based dispatch")
        if re.search(r"^OUT_BOOL\s*=", text, re.MULTILINE):
            failures.append(f"{relative}: fixed OUT_BOOL switch")
        try:
            ast.parse(text, filename=str(relative))
        except SyntaxError as exc:
            failures.append(f"{relative}:{exc.lineno}: {exc.msg}")
            continue
        for name in sorted(undefined_globals(path, text)):
            failures.append(f"{relative}: undefined global {name}")
        for name in sorted(unused_imports(path, text)):
            failures.append(f"{relative}: unused import {name}")
        if path.is_relative_to(SRC / "ntops_lab" / "kernels") and path.name != "__init__.py":
            for name in sorted(unused_kernel_assignments(text)):
                failures.append(f"{relative}: unused module assignment {name}")
    assert not failures, "\n" + "\n".join(failures)


def test_manifest_modules_are_complete_and_importable():
    manifest = json.loads((SRC / "ntops_lab" / "operator_manifest.json").read_text())
    failures = []
    for item in manifest:
        kernel_path = SRC / "ntops_lab" / item["file"]
        spec_path = SRC / "ntops_lab" / "testing" / item["file"].replace("kernels/", "specs/")
        for path, required in (
            (kernel_path, {"run"}),
            (spec_path, {"make_inputs", "run_pytorch", "check"}),
        ):
            if not path.exists():
                failures.append(f"{item['op']}: missing {path.relative_to(ROOT)}")
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
            for name in sorted(required - functions):
                failures.append(f"{path.relative_to(ROOT)}: missing {name}()")
            module_name = "ntops_lab." + str(path.relative_to(SRC / "ntops_lab")).replace("/", ".")[:-3]
            try:
                importlib.import_module(module_name)
            except Exception as exc:
                failures.append(f"{module_name}: {type(exc).__name__}: {exc}")
    assert not failures, "\n" + "\n".join(failures)
