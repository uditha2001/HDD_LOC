from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ddmin_loc_adapter import DDMIN_CORE_PATH, run_original_ddmin_loc
from execution_records import CandidateEvaluation, Outcome, SourceLocation


def test_original_ddmin_is_loaded_from_sibling_research_repository():
    assert DDMIN_CORE_PATH.name == "DeltaMinimize.py"
    assert DDMIN_CORE_PATH.is_file()


def test_original_ddmin_records_chronological_unit_weight_evidence():
    line = SourceLocation("subject.py", 7)

    def evaluator(candidate: str) -> CandidateEvaluation:
        failed = "b" in candidate
        return CandidateEvaluation(
            outcome=Outcome.FAIL if failed else Outcome.PASS,
            coverage=frozenset({line}) if failed else frozenset(),
        )

    result, records = run_original_ddmin_loc("abc", evaluator)

    assert result.minimal_failing_input == "b"
    assert records[0].serialized_input == "abc"
    assert records[0].outcome is Outcome.FAIL
    assert [record.execution_id for record in records] == list(range(len(records)))
    assert all(record.inverse_hdd_weight == 1.0 for record in records)
    assert all(record.tree_depth is None for record in records)


def test_invalid_ddmin_candidate_is_preserved_but_not_interesting():
    def evaluator(candidate: str) -> CandidateEvaluation:
        if candidate == "ab":
            return CandidateEvaluation(outcome=Outcome.FAIL)
        return CandidateEvaluation(
            outcome=Outcome.INVALID,
            structurally_valid=False,
            semantically_valid=False,
        )

    _, records = run_original_ddmin_loc("ab", evaluator)

    invalid = [record for record in records if record.outcome is Outcome.INVALID]
    assert invalid
    assert all(not record.contributes_spectrum_evidence for record in invalid)


def test_ddmin_time_limit_preserves_completed_observations():
    def evaluator(candidate: str) -> CandidateEvaluation:
        time.sleep(0.03)
        return CandidateEvaluation(
            outcome=Outcome.FAIL if "b" in candidate else Outcome.PASS
        )

    result, records = run_original_ddmin_loc(
        "abcdefghij",
        evaluator,
        time_limit_seconds=0.05,
    )

    assert result.timed_out is True
    assert result.time_limit_seconds == 0.05
    assert records
    assert records[0].outcome is Outcome.FAIL
    assert result.minimal_failing_input in {
        record.serialized_input
        for record in records
        if record.outcome is Outcome.FAIL
    }
