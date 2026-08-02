"""Line-coverage replay and static line classification for HDD-LOC.

This module is responsible only for:
  1. bridging HDD test records into per-test line-coverage records
     (collect_line_coverage()), and
  2. statically classifying each line of the program as a predicate,
     a predicate-dependent statement, or an independent statement
     (classify_lines()).

Both are facts about the program (what ran, and how it's structured) --
neither is a suspiciousness score. All SBFL math -- including turning
these facts into per-line scores and blending in predicate context -- lives
in sbfl_score.py. LineCoverageRecord is defined here and imported by
sbfl_score.py so there's exactly one definition of that shape.

Typical flow
------------
    from hdd_loc import HierarchicalDeltaDebugger
    from line_localization import (
        test_cases_from_hdd_result, collect_line_coverage, classify_lines,
    )
    import sbfl

    debugger = HierarchicalDeltaDebugger(oracle=my_oracle)
    hdd_result = debugger.reduce(failing_input)

    test_cases = test_cases_from_hdd_result(hdd_result, debugger)
    coverage = collect_line_coverage("buggy_program.py", test_cases)

    # plain line-level SBFL:
    ranking = sbfl.rank_lines(coverage, formula="ochiai")

    # predicate-context-weighted line-level SBFL:
    classification = classify_lines("buggy_program.py")
    ranking = sbfl.rank_lines_with_predicate_context(coverage, classification)

    print(sbfl.render_line_report("buggy_program.py", ranking))
"""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Dict, List, Optional, Set

from hdd_loc import HDDResult, HierarchicalDeltaDebugger


# --------------------------------------------------------------------------
# Test cases bridged in from HDD
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TestCase:
    """One test case to replay against the buggy program."""

    test_id: int
    candidate: Any
    failed: bool
    weight: float


@dataclass(frozen=True)
class LineCoverageRecord:
    """Per-test line coverage captured from replaying one candidate."""

    test_id: int
    failed: bool
    weight: float
    lines: frozenset[int]


def test_cases_from_hdd_result(
    result: HDDResult,
    debugger: HierarchicalDeltaDebugger,
) -> List[TestCase]:
    """Convert every HDD test record into a replayable line-coverage test
    case, using each record's own WDD weight directly (hdd_loc.py already
    computed the weight that matters -- no separate weighting scheme here).
    """

    if debugger.root is None:
        raise ValueError("debugger has no tree loaded -- call debugger.reduce() first")

    cases: List[TestCase] = []
    for record in result.test_records:
        candidate = debugger._materialize(debugger.root, set(record.active_ids))
        cases.append(
            TestCase(
                test_id=record.test_id,
                candidate=candidate,
                failed=record.failed,
                weight=float(record.weight),
            )
        )
    return cases


# --------------------------------------------------------------------------
# Line tracing
# --------------------------------------------------------------------------


class _LineTracer:
    """Records every line executed within a single target file."""

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
        return False  # never suppress exceptions raised by the target program


def _load_program(program_path: str) -> ModuleType:
    abs_path = os.path.abspath(program_path)
    spec = importlib.util.spec_from_file_location("_buggy_program_under_test", abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load program at {program_path!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_line_coverage(
    program_path: str,
    test_cases: List[TestCase],
    entry_function: str = "run",
) -> List[LineCoverageRecord]:
    """Re-run the buggy program once per test case, recording every line hit.

    A test case whose candidate crashes the program (raises an exception)
    still contributes a partial-coverage record -- the lines executed
    before the crash are exactly the ones relevant to fault localization.
    """

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
                pass  # the exception itself may be how the bug manifests
        records.append(
            LineCoverageRecord(
                test_id=case.test_id,
                failed=case.failed,
                weight=case.weight,
                lines=frozenset(tracer.executed_lines),
            )
        )
    return records


# --------------------------------------------------------------------------
# Static line classification (predicate / dependent / independent)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LineClassification:
    """How one line relates to control flow.

    role: 'predicate' (an if/elif/while/for condition line itself),
          'dependent' (control-dependent on some predicate -- syntactically
          nested inside its body/orelse), or
          'independent' (not nested inside any conditional/loop).
    controlling_predicate: the line number of the predicate this line
        depends on. Equals the line's own number when role == 'predicate'.
        None when role == 'independent'.
    """

    role: str
    controlling_predicate: Optional[int]


def classify_lines(program_path: str) -> Dict[int, LineClassification]:
    """Statically classify every line in program_path.

    This is a syntactic approximation of control dependence -- "nested
    inside a branch/loop's body" -- not a true control-dependence graph.
    That's the right level of precision for weighting purposes here, but
    worth knowing the specific gaps:
      - ternary expressions, comprehension conditions, and boolean
        short-circuit expressions (and/or) are not treated as separate
        predicates; only the enclosing if/elif/while/for/async-for
        statement's own line is.
      - function and class bodies reset the enclosing predicate to None,
        since a nested function's execution isn't control-dependent on
        whatever conditional happened to enclose where it was *defined*.
      - only a statement's first physical line (ast.stmt.lineno) gets
        classified; other physical lines of a multi-line statement have
        no entry, and callers should treat a missing line as unclassified
        rather than assuming 'independent'.
      - elif is handled correctly without special-casing, since Python's
        ast represents each elif as its own nested If inside the parent
        If's orelse -- so each elif's condition becomes its own distinct
        predicate line, with its own body control-dependent on it.
    """

    with open(program_path, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source, filename=program_path)

    classification: Dict[int, LineClassification] = {}

    def record(line: int, enclosing_predicate: Optional[int]) -> None:
        if line in classification:
            return  # first (most specific/innermost) classification wins
        if enclosing_predicate is not None:
            classification[line] = LineClassification(role="dependent", controlling_predicate=enclosing_predicate)
        else:
            classification[line] = LineClassification(role="independent", controlling_predicate=None)

    def classify_block(stmts, enclosing_predicate: Optional[int]) -> None:
        for stmt in stmts:
            classify_stmt(stmt, enclosing_predicate)

    def classify_stmt(stmt: ast.stmt, enclosing_predicate: Optional[int]) -> None:
        line = stmt.lineno

        if isinstance(stmt, (ast.If, ast.While, ast.For, ast.AsyncFor)):
            classification[line] = LineClassification(role="predicate", controlling_predicate=line)
            classify_block(stmt.body, line)
            orelse = getattr(stmt, "orelse", None)
            if orelse:
                classify_block(orelse, line)
            return

        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            record(line, enclosing_predicate)
            classify_block(stmt.body, None)  # fresh scope: not control-dependent on the definition site
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
    from hdd_loc import HierarchicalDeltaDebugger
    import sbfl

    sample_input = {
        "items": [
            {"value": 1},
            {"value": 2},
            {"value": "BUG"},
            {"value": 3},
        ]
    }

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

    base_ranking = sbfl.rank_lines(coverage, formula="ochiai")
    print("--- base line ranking (coverage only) ---")
    print(sbfl.render_line_report(program_path, base_ranking))

    classification = classify_lines(program_path)
    print("\n--- line classification ---")
    for line in sorted(classification):
        info = classification[line]
        print(f"  L{line}: {info.role} (controlling_predicate={info.controlling_predicate})")

    predicate_ranking = sbfl.rank_lines_with_predicate_context(coverage, classification, formula="ochiai")
    print("\n--- predicate-context-weighted line ranking ---")
    print(sbfl.render_line_report(program_path, predicate_ranking))