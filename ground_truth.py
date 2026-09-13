"""Ground-truth extraction used only after suspiciousness ranking."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from execution_records import SourceLocation


_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def deleted_buggy_lines(patch_path: str | Path) -> frozenset[SourceLocation]:
    """Return buggy-side lines removed/replaced by a unified diff."""

    locations: set[SourceLocation] = set()
    filename: str | None = None
    old_line: int | None = None
    for raw_line in Path(patch_path).read_text(encoding="utf-8").splitlines():
        if raw_line.startswith("--- a/"):
            filename = raw_line[6:]
            old_line = None
            continue
        match = _HUNK.match(raw_line)
        if match:
            old_line = int(match.group(1))
            continue
        if filename is None or old_line is None:
            continue
        if raw_line.startswith("-") and not raw_line.startswith("---"):
            locations.add(SourceLocation(filename, old_line))
            old_line += 1
        elif raw_line.startswith("+") and not raw_line.startswith("+++"):
            continue
        elif raw_line.startswith(" "):
            old_line += 1
        elif raw_line.startswith("\\"):
            continue
        else:
            old_line = None
    return frozenset(locations)


def executable_ground_truth(
    locations: frozenset[SourceLocation],
    buggy_root: str | Path,
) -> frozenset[SourceLocation]:
    """Map changed physical lines to their executable Python statement.

    Coverage tools attribute a multi-line assignment or decorator to its first
    executable line.  A patch may change a later continuation line, which can
    never appear in a line spectrum.  This static mapping is evaluation-only:
    it reads the buggy AST and never sees outcomes, scores, or rankings.
    """

    root = Path(buggy_root)
    mapped: set[SourceLocation] = set()
    by_file: dict[str, list[int]] = {}
    for location in locations:
        by_file.setdefault(location.filename, []).append(location.line)

    for filename, changed_lines in by_file.items():
        source_path = root / filename
        try:
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            mapped.update(SourceLocation(filename, line) for line in changed_lines)
            continue

        traceable_lines = {node.lineno for node in ast.walk(tree) if isinstance(node, ast.stmt)}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                traceable_lines.update(decorator.lineno for decorator in node.decorator_list)

        nodes = [
            node for node in ast.walk(tree)
            if hasattr(node, "lineno") and hasattr(node, "end_lineno")
        ]
        for changed_line in changed_lines:
            if changed_line in traceable_lines:
                mapped.add(SourceLocation(filename, changed_line))
                continue
            containing = sorted(
                (
                    node for node in nodes
                    if node.lineno <= changed_line <= node.end_lineno
                ),
                key=lambda node: (node.end_lineno - node.lineno, -node.lineno),
            )
            replacement = None
            for node in containing:
                candidates = [
                    line for line in traceable_lines
                    if node.lineno <= line <= node.end_lineno
                ]
                if candidates:
                    replacement = min(candidates, key=lambda line: (abs(line - changed_line), line))
                    break
            if replacement is None and traceable_lines:
                replacement = min(
                    traceable_lines, key=lambda line: (abs(line - changed_line), line)
                )
            mapped.add(SourceLocation(filename, replacement or changed_line))
    return frozenset(mapped)
