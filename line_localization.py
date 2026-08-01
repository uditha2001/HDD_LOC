"""Line-coverage replay for HDD-LOC.

This module is responsible only for bridging HDD test records into
per-test line-coverage records. It does not perform SBFL scoring.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any, FrozenSet, List, Optional, Set

from hdd_loc import HDDResult, HierarchicalDeltaDebugger


@dataclass(frozen=True)
class TestCase:
    """One test case to replay against the buggy program."""

    test_id: int
    candidate: Any
    failed: bool
    weight: float


def test_cases_from_hdd_result(
    result: HDDResult,
    debugger: HierarchicalDeltaDebugger,
    weight_scheme: str = "linear",
) -> List[TestCase]:
    """Convert HDD test records into replayable line-coverage test cases."""

    if debugger.root is None:
        raise ValueError("debugger has no tree loaded -- call debugger.reduce() first")

    del weight_scheme

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
        return False


def _load_program(program_path: str) -> ModuleType:
    abs_path = os.path.abspath(program_path)
    module_name = f"_buggy_program_under_test_{abs(hash(abs_path))}"
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
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
    """Replay each test case and record the lines executed for that run."""

    target_filename = os.path.abspath(program_path)
    records: List[LineCoverageRecord] = []
    for case in test_cases:
        module = _load_program(program_path)
        entry = getattr(module, entry_function, None)
        if entry is None or not callable(entry):
            raise AttributeError(f"{program_path!r} has no callable {entry_function!r}")

        tracer = _LineTracer(target_filename)
        with tracer:
            try:
                entry(case.candidate)
            except Exception:
                pass
        records.append(
            LineCoverageRecord(
                test_id=case.test_id,
                failed=case.failed,
                weight=case.weight,
                lines=frozenset(tracer.executed_lines),
            )
        )
    return records


if __name__ == "__main__":
    print("line_localization.py now only records per-test line coverage.")
