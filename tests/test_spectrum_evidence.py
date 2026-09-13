from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution_records import ExecutionRecord, Outcome, SourceLocation
from spectrum_evidence import aggregate_observed_spectrum, build_evidence_pool


def _record(execution_id, outcome, coverage, *, valid=True, inverse_hdd_weight=1.0):
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
        structurally_valid=valid,
        semantically_valid=valid,
        runtime_ms=1.0,
        inverse_hdd_weight=inverse_hdd_weight,
    )


def test_exact_spectrum_duplicates_are_counted_once_only_when_requested():
    line10 = SourceLocation("engine.py", 10)
    line20 = SourceLocation("engine.py", 20)
    same_failure_coverage = {line10, line20}
    records = (
        _record(0, Outcome.FAIL, same_failure_coverage),
        _record(1, Outcome.FAIL, same_failure_coverage),
        _record(2, Outcome.FAIL, same_failure_coverage),
        _record(3, Outcome.PASS, {line10}),
    )

    raw = build_evidence_pool(records, deduplicate=False)
    deduplicated = build_evidence_pool(records, deduplicate=True)

    assert len(raw) == 4
    assert len(deduplicated) == 2
    assert deduplicated[0].execution_ids == (0, 1, 2)

    raw_spectrum = aggregate_observed_spectrum(raw)
    deduplicated_spectrum = aggregate_observed_spectrum(deduplicated)
    assert (raw_spectrum[line20].ef, raw_spectrum[line20].ep) == (3, 0)
    assert (deduplicated_spectrum[line20].ef, deduplicated_spectrum[line20].ep) == (1, 0)


def test_budget_is_applied_before_deduplication_and_invalid_filtering():
    line = SourceLocation("parser.py", 9)
    records = (
        _record(0, Outcome.FAIL, {line}),
        _record(1, Outcome.INVALID, set(), valid=False),
        _record(2, Outcome.PASS, {line}),
    )

    evidence = build_evidence_pool(records, deduplicate=True, budget=2)

    assert len(evidence) == 1
    assert evidence[0].outcome is Outcome.FAIL
    assert evidence[0].execution_ids == (0,)


def test_inverse_hdd_weight_is_used_as_fractional_spectrum_evidence():
    line = SourceLocation("engine.py", 20)
    records = (
        _record(0, Outcome.FAIL, {line}, inverse_hdd_weight=0.5),
        _record(1, Outcome.FAIL, set(), inverse_hdd_weight=0.25),
        _record(2, Outcome.PASS, {line}, inverse_hdd_weight=0.125),
        _record(3, Outcome.PASS, set(), inverse_hdd_weight=0.0625),
    )

    weighted = build_evidence_pool(
        records,
        deduplicate=False,
        weighting="inverse_hdd",
    )
    spectrum = aggregate_observed_spectrum(weighted)[line]

    assert (spectrum.ef, spectrum.ep, spectrum.nf, spectrum.np) == (
        0.5,
        0.125,
        0.25,
        0.0625,
    )


def test_deduplicated_inverse_hdd_evidence_uses_group_mean_weight():
    line = SourceLocation("engine.py", 20)
    records = (
        _record(0, Outcome.FAIL, {line}, inverse_hdd_weight=0.5),
        _record(1, Outcome.FAIL, {line}, inverse_hdd_weight=0.25),
    )

    evidence = build_evidence_pool(
        records,
        deduplicate=True,
        weighting="inverse_hdd",
    )

    assert len(evidence) == 1
    assert evidence[0].member_weights == (0.5, 0.25)
    assert evidence[0].evidence_weight == 0.375
