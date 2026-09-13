"""Adapter for running the original DDMIN-LOC reducer in HDD-LOC experiments.

The character-level minimization algorithm is loaded directly from the sibling
``ddmin_loc`` research repository.  This module only adapts its historical
``ask_human`` Boolean-oracle interface to HDD-LOC's semantic evaluator and
canonical execution records.
"""

from __future__ import annotations

import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable, Sequence, Tuple

from execution_records import (
    CandidateEvaluation,
    ExecutionRecord,
    Outcome,
    hash_serialized_input,
)


DDMIN_LOC_ROOT = Path(__file__).resolve().parent.parent / "ddmin_loc"
DDMIN_CORE_PATH = DDMIN_LOC_ROOT / "DeltaMinimize.py"


@dataclass(frozen=True)
class DDMinLOCResult:
    """Original DDMIN result plus its flat generated candidate collections."""

    minimal_failing_input: str
    passing_inputs: Tuple[str, ...]
    failing_inputs: Tuple[str, ...]
    timed_out: bool = False
    time_limit_seconds: float | None = None


@dataclass(frozen=True)
class _Observation:
    serialized_input: str
    evaluation: CandidateEvaluation


class _DDMinTimeLimitReached(Exception):
    """Internal control-flow exception for the experiment time budget."""


@contextmanager
def _wall_clock_limit(seconds: float | None):
    """Interrupt DDMin at its experiment wall-clock limit on Unix.

    HDD-LOC and BugsInPy currently run on Linux. Keeping the timer in this
    adapter prevents the comparison budget from changing the original reducer.
    """

    if seconds is None:
        yield
        return
    if seconds <= 0:
        raise ValueError("DDMin-LOC time limit must be positive")
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("DDMin-LOC wall-clock limiting requires the main thread")

    def handle_timeout(_signum, _frame):
        raise _DDMinTimeLimitReached

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, handle_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def _load_original_ddmin_module() -> ModuleType:
    if not DDMIN_CORE_PATH.is_file():
        raise RuntimeError(
            "original DDMIN-LOC reducer not found at "
            f"{DDMIN_CORE_PATH}; keep ddmin_loc beside HDD_LOC"
        )
    module = ModuleType("_hddloc_original_delta_minimize")
    module.__file__ = str(DDMIN_CORE_PATH)
    source = DDMIN_CORE_PATH.read_bytes()
    exec(compile(source, str(DDMIN_CORE_PATH), "exec"), module.__dict__)
    return module


class _OracleBridge:
    """Expose HDD-LOC's evaluator through DDMIN-LOC's oracle contract."""

    def __init__(self, evaluator: Callable[[str], CandidateEvaluation]) -> None:
        self._evaluator = evaluator
        self.observations: list[_Observation] = []

    @staticmethod
    def _restore_candidate(test_input: Sequence[str] | str) -> str:
        # DeltaMinimize always calls split("_") before ask_human. Rejoining is
        # lossless and keeps underscores inside serialized JSON/source inputs.
        if isinstance(test_input, str):
            return test_input
        return "_".join(str(part) for part in test_input)

    def ask_human(self, test_input: Sequence[str] | str, _program: str) -> bool:
        candidate = self._restore_candidate(test_input)
        started = time.perf_counter()
        evaluation = self._evaluator(candidate)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if evaluation.runtime_ms == 0.0:
            evaluation = CandidateEvaluation(
                outcome=evaluation.outcome,
                coverage=evaluation.coverage,
                structurally_valid=evaluation.structurally_valid,
                semantically_valid=evaluation.semantically_valid,
                runtime_ms=elapsed_ms,
            )
        self.observations.append(_Observation(candidate, evaluation))
        # Historical DDMIN-LOC uses True for PASS and False for FAIL. INVALID
        # must be non-interesting to the reducer, while remaining distinct in
        # the canonical record retained below.
        return evaluation.outcome is not Outcome.FAIL


def _execution_records(observations: Sequence[_Observation]) -> Tuple[ExecutionRecord, ...]:
    return tuple(
        ExecutionRecord(
            execution_id=execution_id,
            parent_id=None,
            input_hash=hash_serialized_input(observation.serialized_input),
            serialized_input=observation.serialized_input,
            active_node_ids=frozenset(),
            tree_depth=None,
            trial_node_id=None,
            removed_node_ids=(),
            removed_subtree_size=None,
            outcome=observation.evaluation.outcome,
            coverage=observation.evaluation.coverage,
            structurally_valid=observation.evaluation.structurally_valid,
            semantically_valid=observation.evaluation.semantically_valid,
            runtime_ms=observation.evaluation.runtime_ms,
            inverse_hdd_weight=1.0,
        )
        for execution_id, observation in enumerate(observations)
    )


def run_original_ddmin_loc(
    failing_input: str,
    evaluator: Callable[[str], CandidateEvaluation],
    *,
    time_limit_seconds: float | None = None,
) -> tuple[DDMinLOCResult, Tuple[ExecutionRecord, ...]]:
    """Run the unmodified DDMIN-LOC character reducer with unit evidence."""

    module = _load_original_ddmin_module()
    oracle = _OracleBridge(evaluator)
    reducer = module.DeltaMinimize("bugsinpy_case", oracle, False)
    timed_out = False
    try:
        with _wall_clock_limit(time_limit_seconds):
            minimal, passing, failing = reducer.find_minimize_input(failing_input)
    except _DDMinTimeLimitReached:
        timed_out = True
        passing = tuple(
            observation.serialized_input
            for observation in oracle.observations
            if observation.evaluation.outcome is Outcome.PASS
        )
        failing = tuple(
            observation.serialized_input
            for observation in oracle.observations
            if observation.evaluation.outcome is Outcome.FAIL
        )
        # The reducer was interrupted, so report the smallest failing input
        # observed so far without claiming that minimization completed.
        minimal = min(failing, key=len) if failing else failing_input
    return (
        DDMinLOCResult(
            minimal_failing_input=minimal,
            passing_inputs=tuple(passing),
            failing_inputs=tuple(failing),
            timed_out=timed_out,
            time_limit_seconds=time_limit_seconds,
        ),
        _execution_records(oracle.observations),
    )
