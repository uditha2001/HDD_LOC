"""Hierarchical Delta Debugging (HDD).

This module provides a small, self-contained HDD implementation for structured
inputs made of nested Python containers such as lists, tuples, dicts, and sets.

The reducer expects a test oracle that returns True when the candidate still
fails. The oracle is intentionally injected so callers can plug in their own
test harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


@dataclass(frozen=True)
class TestRecord:
    """One test attempt produced by HDD."""

    path: Tuple[Union[str, int], ...]
    depth: int
    weight: int
    candidate: Any
    failed: bool


@dataclass(frozen=True)
class HDDResult:
    """Final minimized failing input and all attempted tests."""

    minimal_failing_input: Any
    test_records: List[TestRecord]


class HierarchicalDeltaDebugger:
    """Reduce structured failing input with hierarchical delta debugging."""

    def __init__(self, oracle: Optional[Callable[[Any], bool]] = None) -> None:
        self.oracle = oracle
        self._records: List[TestRecord] = []

    def reduce(self, structured_input: Any) -> HDDResult:
        """Return a minimized failing input and the attempted test cases.

        The oracle must return True for inputs that still fail.
        """

        if self.oracle is None:
            raise NotImplementedError("Provide a test oracle that returns True for failing inputs.")

        self._records = []
        minimal = self._reduce_value(structured_input, path=(), depth=0)
        return HDDResult(minimal_failing_input=minimal, test_records=list(self._records))

    def _record_test(self, candidate: Any, path: Tuple[Union[str, int], ...], depth: int) -> bool:
        failed = bool(self.oracle(candidate))
        self._records.append(
            TestRecord(
                path=path,
                depth=depth,
                weight=depth + 1,
                candidate=candidate,
                failed=failed,
            )
        )
        return failed

    def _reduce_value(self, value: Any, path: Tuple[Union[str, int], ...], depth: int) -> Any:
        if isinstance(value, list):
            return self._reduce_list(value, path, depth)
        if isinstance(value, tuple):
            return tuple(self._reduce_list(list(value), path, depth))
        if isinstance(value, dict):
            return self._reduce_dict(value, path, depth)
        if isinstance(value, set):
            reduced_list = self._reduce_list(sorted(value, key=repr), path, depth)
            return set(reduced_list)
        return value

    def _reduce_list(self, value: List[Any], path: Tuple[Union[str, int], ...], depth: int) -> List[Any]:
        candidate = [self._reduce_value(item, path + (index,), depth + 1) for index, item in enumerate(value)]

        if not self._record_test(candidate, path, depth):
            return candidate

        changed = True
        while changed:
            changed = False
            index = 0
            while index < len(candidate):
                trial = candidate[:index] + candidate[index + 1 :]
                if trial and self._record_test(trial, path, depth):
                    candidate = trial
                    changed = True
                    continue
                index += 1

        return candidate

    def _reduce_dict(self, value: Dict[Any, Any], path: Tuple[Union[str, int], ...], depth: int) -> Dict[Any, Any]:
        candidate: Dict[Any, Any] = {}
        for key, item in value.items():
            candidate[key] = self._reduce_value(item, path + (key,), depth + 1)

        if not self._record_test(candidate, path, depth):
            return candidate

        changed = True
        keys = list(candidate.keys())
        while changed:
            changed = False
            for key in list(keys):
                if key not in candidate:
                    continue
                trial = dict(candidate)
                trial.pop(key)
                if trial and self._record_test(trial, path, depth):
                    candidate = trial
                    keys = list(candidate.keys())
                    changed = True
                    break

        return candidate


def run_hdd(structured_input: Any, oracle: Callable[[Any], bool]) -> HDDResult:
    """Convenience helper for one-off HDD runs."""

    debugger = HierarchicalDeltaDebugger(oracle=oracle)
    return debugger.reduce(structured_input)


def empty_oracle(_: Any) -> bool:
    """Placeholder oracle for callers that want to wire their own test logic later."""

    raise NotImplementedError("Replace empty_oracle with a real test oracle.")
