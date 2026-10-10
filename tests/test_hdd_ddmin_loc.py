from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hdd_ddmin_loc import HDDDDMinLocDebugger


def test_hdd_ddmin_loc_uses_original_ddmin_to_minimize_hdd_siblings():
    seen = []

    def oracle(candidate):
        seen.append(candidate)
        return 3 in candidate

    debugger = HDDDDMinLocDebugger(oracle=oracle)
    result = debugger.reduce([1, 2, 3, 4])

    assert result.minimal_failing_input == [3]
    assert result.test_records[0].failed is True
    assert any(record.failed is False for record in result.test_records)
    # Baseline-HDD evidence semantics are intentionally retained.
    assert all(record.weight == 1.0 for record in result.test_records)
    assert seen


def test_hdd_ddmin_loc_keeps_hdd_tree_reduction_for_nested_input():
    def oracle(candidate):
        return any(2 in item for item in candidate if isinstance(item, list))

    debugger = HDDDDMinLocDebugger(oracle=oracle)

    result = debugger.reduce([[1, 2], 3])

    assert result.minimal_failing_input == [[2]]
    assert result.nodes
    assert any(record.trial_depth is not None for record in result.test_records)
