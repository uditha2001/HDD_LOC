from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bayesian_sbfl import BetaPrior
from execution_records import ExecutionRecord, Outcome, SourceLocation
from main_pipeline import collect_local_hdd_execution_records
from passive_pipeline import analyze_execution_records


def test_local_collect_once_path_returns_filename_qualified_records(tmp_path):
    buggy_path = tmp_path / "buggy_program.py"
    fixed_path = tmp_path / "fixed_program.py"
    buggy_path.write_text(
        "def run(candidate):\n"
        "    total = sum(candidate)\n"
        "    if any(value < 0 for value in candidate):\n"
        "        return total + 1\n"
        "    return total\n",
        encoding="utf-8",
    )
    fixed_path.write_text(
        "def run(candidate):\n"
        "    return sum(candidate)\n",
        encoding="utf-8",
    )

    result, records = collect_local_hdd_execution_records(
        [5, -1, 9],
        str(buggy_path),
        str(fixed_path),
        weighting="uniform",
    )

    assert result.minimal_failing_input == [-1]
    assert len(records) == len(result.test_records)
    assert any(record.outcome is Outcome.FAIL for record in records)
    assert all(
        location.filename == "buggy_program.py"
        for record in records
        for location in record.coverage
    )


def _record(execution_id, outcome, coverage):
    return ExecutionRecord(
        execution_id=execution_id,
        parent_id=None,
        input_hash=str(execution_id),
        serialized_input=str(execution_id),
        active_node_ids=frozenset(),
        tree_depth=None,
        trial_node_id=None,
        removed_node_ids=(),
        removed_subtree_size=None,
        outcome=outcome,
        coverage=frozenset(coverage),
        structurally_valid=True,
        semantically_valid=True,
        runtime_ms=1.0,
        legacy_weight=1.0,
    )


def test_same_records_feed_all_passive_ablation_views_and_budgets():
    line10 = SourceLocation("engine.py", 10)
    line20 = SourceLocation("engine.py", 20)
    records = (
        _record(0, Outcome.FAIL, {line10, line20}),
        _record(1, Outcome.FAIL, {line10, line20}),
        _record(2, Outcome.PASS, {line10}),
    )

    analyses = analyze_execution_records(
        records,
        formulas=("ochiai",),
        budgets=(2, None),
        prior=BetaPrior(),
        monte_carlo_samples=100,
        random_seed=11,
    )

    early, full = analyses
    assert early.original_execution_count == 2
    assert early.raw_evidence_count == 2
    assert early.deduplicated_evidence_count == 1
    assert full.original_execution_count == 3
    assert full.raw_evidence_count == 3
    assert full.deduplicated_evidence_count == 2
    assert full.raw_ranking
    assert full.deduplicated_raw_ranking
    assert full.posterior_mean_ranking
    assert full.monte_carlo_ranking
