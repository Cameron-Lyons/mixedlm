"""Check the Python examples in README.md and docs/ so they cannot drift from the API.

Fences tagged ``python`` run in order in one namespace per Markdown file. Tag
illustrative fragments that are not meant to run (placeholders, ``...``) as
``py``: they render the same way, and like runnable examples they must parse and
their ``mixedlm`` imports and ``mlm.<name>`` references must resolve.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC_FILES = sorted([ROOT / "README.md", *(ROOT / "docs").rglob("*.md")])
FENCE = re.compile(
    r"^(?P<indent>[ \t]*)```(?P<info>[^`\n]*)\n(?P<body>.*?)^(?P=indent)```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
RUNNABLE = "python"
SKETCH = "py"
PACKAGE_ALIASES = {"mixedlm", "mlm"}
OPTIONAL_MODULES = ("matplotlib", "polars", "nlopt")


def _code_blocks(path: Path, *infos: str) -> list[str]:
    text = path.read_text(encoding="utf-8")
    blocks = []
    for match in FENCE.finditer(text):
        if match.group("info").strip() not in infos:
            continue
        indent = match.group("indent")
        lines = [line.removeprefix(indent) for line in match.group("body").splitlines()]
        # Leading newlines make tracebacks and AST nodes report Markdown line numbers.
        blocks.append("\n" * text.count("\n", 0, match.start("body")) + "\n".join(lines))
    return blocks


def _files_with(*infos: str) -> list:
    return [
        pytest.param(path, id=path.relative_to(ROOT).as_posix())
        for path in DOC_FILES
        if _code_blocks(path, *infos)
    ]


def _missing_optional_module(error: ImportError) -> str | None:
    for name in OPTIONAL_MODULES:
        if name in str(error) and importlib.util.find_spec(name) is None:
            return name
    return None


@pytest.mark.parametrize("path", _files_with(RUNNABLE))
def test_examples_run(path, tmp_path, monkeypatch):
    if importlib.util.find_spec("matplotlib") is not None:
        import matplotlib

        matplotlib.use("Agg")
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {"__name__": "__docs__"}
    try:
        for code in _code_blocks(path, RUNNABLE):
            try:
                exec(compile(code, str(path), "exec"), namespace)
            except ImportError as error:
                missing = _missing_optional_module(error)
                if missing is None:
                    raise
                pytest.skip(f"optional dependency {missing} is not installed")
    finally:
        if "matplotlib.pyplot" in sys.modules:
            sys.modules["matplotlib.pyplot"].close("all")


def _unresolved_references(code: str) -> list[str]:
    package = importlib.import_module("mixedlm")
    unresolved = []
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "mixedlm":
            if importlib.util.find_spec(node.module) is None:
                unresolved.append(f"line {node.lineno}: module {node.module}")
                continue
            module = importlib.import_module(node.module)
            for alias in node.names:
                submodule = f"{node.module}.{alias.name}"
                if not hasattr(module, alias.name) and importlib.util.find_spec(submodule) is None:
                    unresolved.append(f"line {node.lineno}: {submodule}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root == "mixedlm" and importlib.util.find_spec(alias.name) is None:
                    unresolved.append(f"line {node.lineno}: module {alias.name}")
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in PACKAGE_ALIASES
            and not hasattr(package, node.attr)
        ):
            unresolved.append(f"line {node.lineno}: {node.value.id}.{node.attr}")
    return unresolved


@pytest.mark.parametrize("path", _files_with(RUNNABLE, SKETCH))
def test_references_resolve(path):
    unresolved = [
        reference
        for code in _code_blocks(path, RUNNABLE, SKETCH)
        for reference in _unresolved_references(code)
    ]
    assert not unresolved, f"{path.relative_to(ROOT)}: {unresolved}"
