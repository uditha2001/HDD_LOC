from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution_records import (
    CandidateEvaluation,
    Outcome,
    RecordingOracle,
    SourceLocation,
    build_execution_records,
    read_execution_records_jsonl,
    write_execution_records_jsonl,
)
from hdd_algorithm import HierarchicalDeltaDebugger


def test_recording_oracle_preserves_every_hdd_execution():
    location = SourceLocation("engine.py", 12)

    def evaluator(candidate):
        failed = "bug" in candidate
        return CandidateEvaluation(
            outcome=Outcome.FAIL if failed else Outcome.PASS,
            coverage=frozenset({location}),
        )

    oracle = RecordingOracle(evaluator)
    debugger = HierarchicalDeltaDebugger(oracle=oracle, weighting="uniform")
    result = debugger.reduce(["noise", "bug"])
    records = build_execution_records(result, oracle.observations)

    assert result.minimal_failing_input == ["bug"]
    assert len(records) == len(result.test_records) == len(oracle.observations)
    assert [record.execution_id for record in records] == list(range(len(records)))
    assert all(record.coverage == frozenset({location}) for record in records)
    assert all(len(record.input_hash) == 64 for record in records)
    # The tested two-leaf partition carries HDD's recorded 1 / weight.
    assert records[1].inverse_hdd_weight == 0.5


def test_invalid_candidate_is_distinct_from_pass_but_not_interesting():
    oracle = RecordingOracle(
        lambda candidate: CandidateEvaluation(
            outcome=Outcome.INVALID,
            structurally_valid=False,
            semantically_valid=False,
        )
    )

    assert oracle(["broken"]) is False
    assert oracle.observations[0].evaluation.outcome is Outcome.INVALID


def test_execution_records_round_trip_as_jsonl(tmp_path):
    oracle = RecordingOracle(
        lambda candidate: CandidateEvaluation(
            outcome=Outcome.FAIL,
            coverage=frozenset({SourceLocation("parser.py", 93)}),
        )
    )
    debugger = HierarchicalDeltaDebugger(oracle=oracle, weighting="uniform")
    result = debugger.reduce(["bug"])
    records = build_execution_records(result, oracle.observations)
    output_path = tmp_path / "executions.jsonl"

    write_execution_records_jsonl(records, output_path)
    loaded = read_execution_records_jsonl(output_path)

    assert loaded == records
