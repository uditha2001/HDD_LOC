#!/usr/bin/env python3
"""Instrument a BugsInPy Python module for predicate-outcome collection.

The original DDMIN-LOC Codeflaws helper instruments C control-flow
conditions as ``LOG_PRED(id, line, condition)``.  This tool supplies the
equivalent *source-to-source* transformation for a BugsInPy Python module.

Only ``if`` and ``while`` tests are predicates here.  Python ``for`` loops do
not have a Boolean condition comparable to C's middle ``for`` expression, so
they are intentionally not instrumented.  The transformed expression
evaluates the original test once, records its Boolean outcome, and returns
that Boolean value to the control-flow statement.

Set ``HDD_LOC_PREDICATE_LOG`` when executing the generated module.  Every
evaluation appends one compatible record, for example::

    LOG_PRED called: id=2 line=41 cond=1

The line is the original source line.  The transformation makes no inserted
top-level imports or helper definitions, preserving physical source-line
positions for coverage collection.

This is an instrumentation generator only.  It neither runs BugsInPy tests
nor changes HDD reduction or localization scoring.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class PredicateSite:
    """Stable identity and original location of one instrumented test."""

    predicate_id: int
    line: int
    column: int
    kind: str
    expression: str


@dataclass(frozen=True)
class InstrumentationResult:
    source: str
    predicates: tuple[PredicateSite, ...]


def _absolute_offset(source: str, line: int, utf8_column: int) -> int:
    """Convert an AST UTF-8 byte column into a Python string offset."""

    if line <= 0:
        raise ValueError("AST source lines must be positive")
    lines = source.splitlines(keepends=True)
    if line > len(lines):
        raise ValueError("AST source line is outside the input")
    prefix = sum(len(item) for item in lines[: line - 1])
    encoded_line = lines[line - 1].encode("utf-8")
    try:
        decoded_prefix = encoded_line[:utf8_column].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("AST column splits a UTF-8 character") from exc
    return prefix + len(decoded_prefix)


def _predicate_expression(predicate_id: int, line: int, expression: str) -> str:
    """Return an expression that evaluates ``expression`` exactly once.

    A self-contained expression avoids inserting an import or runtime helper
    into the target module.  That is important because shifting the target
    file's lines would make its coverage locations no longer match the
    original BugsInPy source.  Logging is disabled unless the environment
    variable is configured.
    """

    record = "LOG_PRED called: id={0} line={1} cond={{0}}".format(predicate_id, line)
    return (
        "(lambda __hdd_value: "
        "(lambda __hdd_outcome, __hdd_log: "
        "((print(" + repr(record) + ".format(int(__hdd_outcome)), "
        "file=open(__hdd_log, 'a', encoding='utf-8')) if __hdd_log else None), "
        "__hdd_outcome)[1]"
        ")(bool(__hdd_value), __import__('os').environ.get('HDD_LOC_PREDICATE_LOG'))"
        ")(" + expression + ")"
    )


def _iter_predicate_nodes(tree: ast.AST) -> Iterable[tuple[str, ast.expr]]:
    """Yield control-flow tests in source order without overlapping edits."""

    sites: list[tuple[str, ast.expr]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            sites.append(("if", node.test))
        elif isinstance(node, ast.While):
            sites.append(("while", node.test))
    return sorted(sites, key=lambda item: (item[1].lineno, item[1].col_offset))


def instrument_source(source: str, filename: str = "<source>") -> InstrumentationResult:
    """Instrument Python ``if`` and ``while`` tests while retaining line layout."""

    tree = ast.parse(source, filename=filename)
    replacements: list[tuple[int, int, str]] = []
    predicates: list[PredicateSite] = []
    for predicate_id, (kind, test) in enumerate(_iter_predicate_nodes(tree), start=1):
        if not hasattr(test, "end_lineno") or test.end_lineno is None:
            raise ValueError("Python parser did not provide expression end positions")
        start = _absolute_offset(source, test.lineno, test.col_offset)
        end = _absolute_offset(source, test.end_lineno, test.end_col_offset)
        expression = source[start:end]
        if not expression.strip():
            continue
        replacements.append((start, end, _predicate_expression(predicate_id, test.lineno, expression)))
        predicates.append(
            PredicateSite(
                predicate_id=predicate_id,
                line=test.lineno,
                column=test.col_offset,
                kind=kind,
                expression=expression,
            )
        )

    instrumented = source
    for start, end, replacement in sorted(replacements, reverse=True):
        instrumented = instrumented[:start] + replacement + instrumented[end:]
    return InstrumentationResult(source=instrumented, predicates=tuple(predicates))


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(input_path.stem + "_instrumented" + input_path.suffix)


def _write_manifest(path: Path, predicates: Sequence[PredicateSite]) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "predicate_id": site.predicate_id,
                    "line": site.line,
                    "column": site.column,
                    "kind": site.kind,
                    "expression": site.expression,
                }
                for site in predicates
            ],
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="BugsInPy Python file to instrument")
    parser.add_argument("--output", type=Path, help="generated module path")
    parser.add_argument("--manifest", type=Path, help="predicate ID-to-source-location JSON path")
    args = parser.parse_args(argv)

    source_path = args.source.resolve()
    if not source_path.is_file():
        parser.error("source is not a file: {0}".format(source_path))
    output_path = (args.output or _default_output_path(source_path)).resolve()
    if output_path == source_path:
        parser.error("output must differ from source; keep the original module unchanged")

    result = instrument_source(source_path.read_text(encoding="utf-8"), str(source_path))
    output_path.write_text(result.source, encoding="utf-8")
    manifest_path = (args.manifest or output_path.with_suffix(".predicates.json")).resolve()
    _write_manifest(manifest_path, result.predicates)
    print("instrumented {0} predicates -> {1}".format(len(result.predicates), output_path))
    print("predicate manifest -> {0}".format(manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
