"""Collect-once differential executor for local Python program pairs."""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Optional, Set

from execution_records import CandidateEvaluation, Outcome, SourceLocation


@dataclass(frozen=True)
class _CallResult:
    kind: str
    value: Any = None
    exception_type: Optional[str] = None
    message: Optional[str] = None


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


def _load_program(program_path: str, module_name: str) -> ModuleType:
    absolute = os.path.abspath(program_path)
    spec = importlib.util.spec_from_file_location(module_name, absolute)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load program at {program_path!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _capture_call(entry, candidate: Any, tracer: Optional[_LineTracer] = None) -> _CallResult:
    try:
        if tracer is None:
            value = entry(candidate)
        else:
            with tracer:
                value = entry(candidate)
        return _CallResult(kind="value", value=value)
    except Exception as exc:
        return _CallResult(
            kind="exception",
            exception_type=type(exc).__qualname__,
            message=str(exc),
        )


class LocalDifferentialAdapter:
    """Evaluate fresh buggy/fixed modules and trace only the buggy version."""

    def __init__(
        self,
        buggy_program: str,
        fixed_program: str,
        entry_function: str = "run",
        coverage_filename: Optional[str] = None,
    ) -> None:
        self.buggy_program = os.path.abspath(buggy_program)
        self.fixed_program = os.path.abspath(fixed_program)
        self.entry_function = entry_function
        self.coverage_filename = coverage_filename or Path(self.buggy_program).name

        # Fail fast on repository/setup errors rather than labeling them as
        # candidate invalidity.
        self._entry(
            _load_program(self.buggy_program, "_hddloc_buggy_validation"),
            self.buggy_program,
        )
        self._entry(
            _load_program(self.fixed_program, "_hddloc_fixed_validation"),
            self.fixed_program,
        )

    def _entry(self, module: ModuleType, path: str):
        entry = getattr(module, self.entry_function, None)
        if entry is None or not callable(entry):
            raise AttributeError(f"{path!r} has no callable {self.entry_function!r}")
        return entry

    def evaluate(self, candidate: Any) -> CandidateEvaluation:
        started = time.perf_counter()
        # Fresh modules prevent candidate-to-candidate global-state leakage.
        buggy_module = _load_program(self.buggy_program, "_hddloc_buggy_execution")
        fixed_module = _load_program(self.fixed_program, "_hddloc_fixed_execution")
        buggy_entry = self._entry(buggy_module, self.buggy_program)
        fixed_entry = self._entry(fixed_module, self.fixed_program)

        tracer = _LineTracer(self.buggy_program)
        buggy = _capture_call(buggy_entry, candidate, tracer)
        fixed = _capture_call(fixed_entry, candidate)

        if buggy.kind != fixed.kind:
            outcome = Outcome.FAIL
            semantically_valid = True
        elif buggy.kind == "value":
            outcome = Outcome.FAIL if buggy.value != fixed.value else Outcome.PASS
            semantically_valid = True
        elif (
            buggy.exception_type != fixed.exception_type
            or buggy.message != fixed.message
        ):
            outcome = Outcome.FAIL
            semantically_valid = True
        else:
            # Equal exceptions do not provide a semantic buggy/fixed contrast.
            outcome = Outcome.INVALID
            semantically_valid = False

        coverage = frozenset(
            SourceLocation(self.coverage_filename, line)
            for line in tracer.executed_lines
        )
        return CandidateEvaluation(
            outcome=outcome,
            coverage=coverage,
            structurally_valid=True,
            semantically_valid=semantically_valid,
            runtime_ms=(time.perf_counter() - started) * 1000.0,
        )
