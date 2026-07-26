"""Depth-Weighted Line-Level Fault Localization.

This module takes:
  1. a "buggy program" -- a Python source file exposing an entry function
     that accepts one argument (the candidate structured input), and
  2. the list of test cases HDD already ran (from hdd_loc.HDDResult) --

and, for every one of those test cases, re-executes the program against
that test case's candidate input while tracing exactly which source lines
run. Each test case also carries a weight (derived from the depth of the
input node HDD was toggling when it produced that test), so the resulting
per-line coverage records let you compute *depth-weighted* SBFL
suspiciousness scores per line, not just per input-structure node.

This is the second half of HDD-LOC: hdd_loc.py localizes suspicious
*input* structure; this module localizes suspicious *program* lines, using
the same test cases and the same depth-weighting idea.

Typical flow
------------
    from hdd_loc import HierarchicalDeltaDebugger
    from line_localization import (
        test_cases_from_hdd_result, collect_line_coverage,
        aggregate_line_spectrum, rank_lines, render_report,
    )

    debugger = HierarchicalDeltaDebugger(oracle=my_oracle)
    hdd_result = debugger.reduce(failing_input)

    test_cases = test_cases_from_hdd_result(hdd_result, debugger)
    coverage = collect_line_coverage("buggy_program.py", test_cases)
    spectrum = aggregate_line_spectrum(coverage)
    ranking = rank_lines(spectrum, formula="ochiai")

    print(render_report("buggy_program.py", ranking))
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

# hdd_loc.py is expected to sit alongside this file.
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


def depth_weight(depth: int, max_depth: int, scheme: str = "linear") -> float:
    """Same weighting schemes as hdd_loc, kept local so this module has no
    dependency on hdd_loc's private helpers.
    """

    if scheme == "linear":
        return depth + 1
    if scheme == "normalized":
        return (depth + 1) / (max_depth + 1)
    if scheme == "exponential":
        return 2.0 ** depth
    if scheme == "log":
        return math.log2(depth + 2)
    raise ValueError(f"unknown weight scheme: {scheme!r}")


def test_cases_from_hdd_result(
    result: HDDResult,
    debugger: HierarchicalDeltaDebugger,
    weight_scheme: str = "linear",
) -> List[TestCase]:
    """Turn every TestRecord HDD produced into a TestCase for line coverage.

    ``debugger`` must be the same HierarchicalDeltaDebugger instance that
    produced ``result`` (its ``root`` is used to re-materialize each test's
    candidate from the record's active node ids). A test's weight is the
    depth weight of the node HDD was trying to remove when it ran that
    test; the baseline (whole-input) test is weighted as depth 0.
    """

    if debugger.root is None:
        raise ValueError("debugger has no tree loaded -- call debugger.reduce() first")

    max_depth = max((node.depth for node in result.nodes.values()), default=0)

    cases: List[TestCase] = []
    for record in result.test_records:
        candidate = debugger._materialize(debugger.root, set(record.active_ids))
        depth = record.trial_depth if record.trial_depth is not None else 0
        weight = depth_weight(depth, max_depth, weight_scheme)
        cases.append(TestCase(test_id=record.test_id, candidate=candidate, failed=record.failed, weight=weight))
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
            # Don't trace into library/stdlib frames; the global trace
            # function is still invoked for their nested calls, so this
            # only restricts *line* recording, not visibility of further
            # calls back into the target file.
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


@dataclass(frozen=True)
class LineCoverageRecord:
    """Which lines ran for one test case, and that test's outcome/weight."""

    test_id: int
    failed: bool
    weight: float
    lines: FrozenSet[int]


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
# Weighted spectrum and suspiciousness
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LineSpectrum:
    """Weighted spectrum counts for one source line."""

    line: int
    ef: float  # weighted sum of failing tests that executed this line
    ep: float  # weighted sum of passing tests that executed this line
    nf: float  # weighted sum of failing tests that did NOT execute this line
    np: float  # weighted sum of passing tests that did NOT execute this line


def aggregate_line_spectrum(records: List[LineCoverageRecord]) -> Dict[int, LineSpectrum]:
    """Build the weighted spectrum for every line touched by any test."""

    all_lines: Set[int] = set()
    for record in records:
        all_lines |= record.lines

    total_failed_weight = sum(r.weight for r in records if r.failed)
    total_passed_weight = sum(r.weight for r in records if not r.failed)

    spectrum: Dict[int, LineSpectrum] = {}
    for line in all_lines:
        ef = sum(r.weight for r in records if r.failed and line in r.lines)
        ep = sum(r.weight for r in records if not r.failed and line in r.lines)
        nf = total_failed_weight - ef
        np_ = total_passed_weight - ep
        spectrum[line] = LineSpectrum(line=line, ef=ef, ep=ep, nf=nf, np=np_)
    return spectrum


def _sbfl_score(ef: float, ep: float, nf: float, np_: float, formula: str) -> float:
    if formula == "tarantula":
        fail_ratio = ef / (ef + nf) if (ef + nf) else 0.0
        pass_ratio = ep / (ep + np_) if (ep + np_) else 0.0
        denom = fail_ratio + pass_ratio
        return fail_ratio / denom if denom else 0.0
    if formula == "ochiai":
        denom = math.sqrt((ef + nf) * (ef + ep))
        return ef / denom if denom else 0.0
    if formula == "jaccard":
        denom = ef + nf + ep
        return ef / denom if denom else 0.0
    raise ValueError(f"unknown SBFL formula: {formula!r}")


def rank_lines(
    spectrum: Dict[int, LineSpectrum],
    formula: str = "ochiai",
    normalize: bool = True,
) -> List[Tuple[int, float, LineSpectrum]]:
    """Score every line and return (line, score, spectrum) sorted descending."""

    scored = [(line, _sbfl_score(s.ef, s.ep, s.nf, s.np, formula), s) for line, s in spectrum.items()]
    if normalize and scored:
        max_score = max(score for _, score, _ in scored) or 1.0
        scored = [(line, score / max_score, s) for line, score, s in scored]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def render_report(program_path: str, ranking: List[Tuple[int, float, LineSpectrum]], top_n: Optional[int] = None) -> str:
    """Render a human-readable, source-annotated suspiciousness report."""

    with open(program_path, "r", encoding="utf-8") as f:
        source_lines = f.readlines()

    ranked = ranking if top_n is None else ranking[:top_n]
    rows = []
    for line_no, score, spec in ranked:
        text = source_lines[line_no - 1].rstrip("\n") if 0 < line_no <= len(source_lines) else ""
        rows.append(
            f"  {score:6.3f}  L{line_no:<4} ef={spec.ef:g} ep={spec.ep:g} "
            f"nf={spec.nf:g} np={spec.np:g}   {text}"
        )
    header = f"Suspiciousness report for {program_path}"
    return header + "\n" + "\n".join(rows)


if __name__ == "__main__":
    # Demo: reuse hdd_loc's reduction on a small nested input, bridge its
    # test cases into this module, and localize the actual buggy line in
    # buggy_sample.py.
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

    test_cases = test_cases_from_hdd_result(hdd_result, debugger, weight_scheme="linear")
    print(f"Bridged {len(test_cases)} HDD test cases into line-coverage test cases")

    coverage = collect_line_coverage(program_path, test_cases)
    spectrum = aggregate_line_spectrum(coverage)
    ranking = rank_lines(spectrum, formula="ochiai")

    print()
    print(render_report(program_path, ranking))