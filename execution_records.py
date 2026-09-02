"""Canonical execution records for HDD-LOC.

The HDD reducer has its own structural trial records in ``hdd_algorithm``.
This module defines the richer, localization-facing record produced when a
candidate is actually evaluated against the buggy and reference programs.

Original execution records are append-only evidence.  Spectrum
deduplication is deliberately implemented elsewhere so ancestry and
diagnostic information are never discarded.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, FrozenSet, Iterable, Optional, Sequence, Tuple


class Outcome(str, Enum):
    """Semantic result of evaluating one HDD candidate."""

    FAIL = "FAIL"
    PASS = "PASS"
    INVALID = "INVALID"


@dataclass(frozen=True, order=True)
class SourceLocation:
    """A source line in the buggy program, qualified by relative filename."""

    filename: str
    line: int

    def __post_init__(self) -> None:
        normalized = self.filename.replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        if not normalized:
            raise ValueError("source filename must not be empty")
        if self.line <= 0:
            raise ValueError("source line must be positive")
        object.__setattr__(self, "filename", normalized)


@dataclass(frozen=True)
class CandidateEvaluation:
    """Rich result returned by a program-specific candidate evaluator."""

    outcome: Outcome
    coverage: FrozenSet[SourceLocation] = frozenset()
    structurally_valid: bool = True
    semantically_valid: bool = True
    runtime_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.outcome is Outcome.INVALID and self.structurally_valid and self.semantically_valid:
            raise ValueError("INVALID evaluation must identify structural or semantic invalidity")
        if self.runtime_ms < 0:
            raise ValueError("runtime_ms must not be negative")


@dataclass(frozen=True)
class OracleObservation:
    """One evaluator result captured by :class:`RecordingOracle`."""

    input_hash: str
    serialized_input: str
    evaluation: CandidateEvaluation


@dataclass(frozen=True)
class ExecutionRecord:
    """Canonical, lossless record for one HDD candidate execution."""

    execution_id: int
    parent_id: Optional[int]
    input_hash: str
    serialized_input: str
    active_node_ids: FrozenSet[int]
    tree_depth: Optional[int]
    trial_node_id: Optional[int]
    removed_node_ids: Tuple[int, ...]
    removed_subtree_size: Optional[int]
    outcome: Outcome
    coverage: FrozenSet[SourceLocation]
    structurally_valid: bool
    semantically_valid: bool
    runtime_ms: float
    legacy_weight: float

    @property
    def contributes_spectrum_evidence(self) -> bool:
        return (
            self.outcome in (Outcome.FAIL, Outcome.PASS)
            and self.structurally_valid
            and self.semantically_valid
        )


def _canonical_value(value: Any) -> Any:
    """Convert common structured-input values into stable JSON data."""

    if isinstance(value, dict):
        items = [(_canonical_value(key), _canonical_value(item)) for key, item in value.items()]
        items.sort(key=lambda pair: json.dumps(pair[0], sort_keys=True, default=repr))
        return {"__type__": "dict", "items": items}
    if isinstance(value, list):
        return {"__type__": "list", "items": [_canonical_value(item) for item in value]}
    if isinstance(value, tuple):
        return {"__type__": "tuple", "items": [_canonical_value(item) for item in value]}
    if isinstance(value, set):
        items = [_canonical_value(item) for item in value]
        items.sort(key=lambda item: json.dumps(item, sort_keys=True, default=repr))
        return {"__type__": "set", "items": items}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return {"__type__": type(value).__qualname__, "repr": repr(value)}


def serialize_structured_input(candidate: Any) -> str:
    """Stable default serialization used for candidate identity."""

    return json.dumps(
        _canonical_value(candidate),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def hash_serialized_input(serialized_input: str) -> str:
    return hashlib.sha256(serialized_input.encode("utf-8")).hexdigest()


class RecordingOracle:
    """Adapt a rich evaluator to HDD's existing Boolean oracle contract.

    Every call is retained, including duplicate candidates and INVALID
    candidates.  INVALID is returned to HDD as non-interesting while its
    distinct outcome remains available in ``observations``.
    """

    def __init__(
        self,
        evaluator: Callable[[Any], CandidateEvaluation],
        serializer: Callable[[Any], str] = serialize_structured_input,
    ) -> None:
        self.evaluator = evaluator
        self.serializer = serializer
        self.observations: list[OracleObservation] = []

    def __call__(self, candidate: Any) -> bool:
        serialized = self.serializer(candidate)
        started = time.perf_counter()
        evaluation = self.evaluator(candidate)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if evaluation.runtime_ms == 0.0:
            evaluation = CandidateEvaluation(
                outcome=evaluation.outcome,
                coverage=evaluation.coverage,
                structurally_valid=evaluation.structurally_valid,
                semantically_valid=evaluation.semantically_valid,
                runtime_ms=elapsed_ms,
            )
        self.observations.append(
            OracleObservation(
                input_hash=hash_serialized_input(serialized),
                serialized_input=serialized,
                evaluation=evaluation,
            )
        )
        return evaluation.outcome is Outcome.FAIL


def build_execution_records(
    hdd_result: Any,
    observations: Sequence[OracleObservation],
) -> Tuple[ExecutionRecord, ...]:
    """Join HDD structural trials with same-execution oracle observations."""

    trials = hdd_result.test_records
    if len(trials) != len(observations):
        raise ValueError(
            "HDD trial/observation count mismatch: "
            f"{len(trials)} structural trials versus {len(observations)} observations"
        )

    records = []
    for trial, observation in zip(trials, observations):
        evaluation = observation.evaluation
        expected_failed = evaluation.outcome is Outcome.FAIL
        if bool(trial.failed) != expected_failed:
            raise ValueError(
                f"outcome mismatch for execution {trial.test_id}: "
                f"HDD failed={trial.failed!r}, evaluator outcome={evaluation.outcome.value}"
            )
        records.append(
            ExecutionRecord(
                execution_id=int(trial.test_id),
                parent_id=None,
                input_hash=observation.input_hash,
                serialized_input=observation.serialized_input,
                active_node_ids=frozenset(trial.active_ids),
                tree_depth=trial.trial_depth,
                trial_node_id=trial.trial_node_id,
                removed_node_ids=(),
                removed_subtree_size=None,
                outcome=evaluation.outcome,
                coverage=evaluation.coverage,
                structurally_valid=evaluation.structurally_valid,
                semantically_valid=evaluation.semantically_valid,
                runtime_ms=evaluation.runtime_ms,
                legacy_weight=float(trial.weight),
            )
        )
    return tuple(records)


def _record_to_json(record: ExecutionRecord) -> dict[str, Any]:
    return {
        "execution_id": record.execution_id,
        "parent_id": record.parent_id,
        "input_hash": record.input_hash,
        "serialized_input": record.serialized_input,
        "active_node_ids": sorted(record.active_node_ids),
        "tree_depth": record.tree_depth,
        "trial_node_id": record.trial_node_id,
        "removed_node_ids": list(record.removed_node_ids),
        "removed_subtree_size": record.removed_subtree_size,
        "outcome": record.outcome.value,
        "coverage": [asdict(location) for location in sorted(record.coverage)],
        "structurally_valid": record.structurally_valid,
        "semantically_valid": record.semantically_valid,
        "runtime_ms": record.runtime_ms,
        "legacy_weight": record.legacy_weight,
    }


def write_execution_records_jsonl(records: Iterable[ExecutionRecord], output_path: str | Path) -> None:
    """Persist original records without applying evidence deduplication."""

    path = Path(output_path)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(_record_to_json(record), sort_keys=True, ensure_ascii=False))
            stream.write("\n")


def read_execution_records_jsonl(input_path: str | Path) -> Tuple[ExecutionRecord, ...]:
    """Load a previously collected execution suite without rerunning HDD."""

    records = []
    path = Path(input_path)
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                record = ExecutionRecord(
                    execution_id=int(item["execution_id"]),
                    parent_id=item["parent_id"],
                    input_hash=str(item["input_hash"]),
                    serialized_input=str(item["serialized_input"]),
                    active_node_ids=frozenset(
                        int(node_id) for node_id in item["active_node_ids"]
                    ),
                    tree_depth=item["tree_depth"],
                    trial_node_id=item["trial_node_id"],
                    removed_node_ids=tuple(int(node_id) for node_id in item["removed_node_ids"]),
                    removed_subtree_size=item["removed_subtree_size"],
                    outcome=Outcome(item["outcome"]),
                    coverage=frozenset(
                        SourceLocation(location["filename"], int(location["line"]))
                        for location in item["coverage"]
                    ),
                    structurally_valid=bool(item["structurally_valid"]),
                    semantically_valid=bool(item["semantically_valid"]),
                    runtime_ms=float(item["runtime_ms"]),
                    legacy_weight=float(item["legacy_weight"]),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid execution record at {path}:{line_number}") from exc
            records.append(record)
    return tuple(records)
