"""Line coverage replay for the HDD baseline (unweighted).

This module provides the small adapter that turns HDD baseline test
records into per-test line-coverage records suitable for unweighted SBFL.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Dict, List, Optional, Set

from hdd_baseline import HDDResult, HierarchicalDeltaDebugger


@dataclass(frozen=True)
class TestCase:
    test_id: int
    candidate: Any
    failed: bool


@dataclass(frozen=True)
class LineCoverageRecord:
    test_id: int
    failed: bool
    lines: frozenset[int]


def test_cases_from_hdd_result(result: HDDResult, debugger: HierarchicalDeltaDebugger) -> List[TestCase]:
    if debugger.root is None:
        raise ValueError("debugger has no tree loaded -- call debugger.reduce() first")

    cases: List[TestCase] = []
    for record in result.test_records:
        candidate = debugger._materialize(debugger.root, set(record.active_ids))
        cases.append(TestCase(test_id=record.test_id, candidate=candidate, failed=record.failed))
    return cases


class _LineTracer:
    def __init__(self, target_filename: str) -> None:
        self.target_filename = target_filename
        self.executed_lines: Set[int] = set()
        self._previous_trace = None

    def _trace(self, frame, event, arg):
        if frame.f_code.co_filename != self.target_filename:
            return None
        if event == "line":
            self.executed_lines.add(frame.f_lineno)
        return self._trace

    def __enter__(self) -> "_LineTracer":
        self.executed_lines = set()
        self._previous_trace = sys.gettrace()
        sys.settrace(self._trace)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        sys.settrace(self._previous_trace)
        return False


def _load_program(program_path: str) -> ModuleType:
    abs_path = os.path.abspath(program_path)
    spec = importlib.util.spec_from_file_location("_buggy_program_under_test", abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load program at {program_path!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_line_coverage(program_path: str, test_cases: List[TestCase], entry_function: str = "run") -> List[LineCoverageRecord]:
    target_filename = os.path.abspath(program_path)
    module = _load_program(program_path)
    entry = getattr(module, entry_function, None)
    if entry is None or not callable(entry):
        raise AttributeError(f"{program_path!r} has no callable {entry_function!r}")

    records: List[LineCoverageRecord] = []
    for case in test_cases:
        tracer = _LineTracer(target_filename)
        with tracer:
            try:
                entry(case.candidate)
            except Exception:
                pass
        records.append(LineCoverageRecord(test_id=case.test_id, failed=case.failed, lines=frozenset(tracer.executed_lines)))
    return records


def classify_lines(program_path: str) -> Dict[int, str]:
    # Reuse the simple predicate classification from weighted module if desired.
    with open(program_path, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source, filename=program_path)
    classification: Dict[int, str] = {}

    def record(line: int, enclosing_predicate: Optional[int]) -> None:
        if line in classification:
            return
        classification[line] = "dependent" if enclosing_predicate is not None else "independent"

    def classify_block(stmts, enclosing_predicate: Optional[int]) -> None:
        for stmt in stmts:
            classify_stmt(stmt, enclosing_predicate)

    def classify_stmt(stmt: ast.stmt, enclosing_predicate: Optional[int]) -> None:
        line = stmt.lineno
        if isinstance(stmt, (ast.If, ast.While, ast.For, ast.AsyncFor)):
            classification[line] = "predicate"
            classify_block(stmt.body, line)
            orelse = getattr(stmt, "orelse", None)
            if orelse:
                classify_block(orelse, line)
            return
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            record(line, enclosing_predicate)
            classify_block(stmt.body, None)
            return
        if isinstance(stmt, ast.Try):
            record(line, enclosing_predicate)
            classify_block(stmt.body, enclosing_predicate)
            for handler in stmt.handlers:
                classify_block(handler.body, enclosing_predicate)
            if stmt.orelse:
                classify_block(stmt.orelse, enclosing_predicate)
            if stmt.finalbody:
                classify_block(stmt.finalbody, enclosing_predicate)
            return
        if isinstance(stmt, (ast.With, ast.AsyncWith)):
            record(line, enclosing_predicate)
            classify_block(stmt.body, enclosing_predicate)
            return
        record(line, enclosing_predicate)

    classify_block(tree.body, None)
    return classification


if __name__ == "__main__":
    from hdd_baseline import HierarchicalDeltaDebugger

    sample_input = {"items": [{"value": 1}, {"value": 2}, {"value": "BUG"}, {"value": 3}]}
    program_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "buggy_sample.py")
    program_module = _load_program(program_path)

    def oracle(candidate: Any) -> bool:
        try:
            program_module.run(candidate)
            return False
        except Exception:
            return True

    debugger = HierarchicalDeltaDebugger(oracle=oracle)
    hdd_result = debugger.reduce(sample_input)
    print("Minimal failing input:", hdd_result.minimal_failing_input)

    test_cases = test_cases_from_hdd_result(hdd_result, debugger)
    coverage = collect_line_coverage(program_path, test_cases)
    print(f"Collected {len(coverage)} coverage records")
