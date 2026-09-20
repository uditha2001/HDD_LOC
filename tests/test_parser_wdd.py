from __future__ import annotations

import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser_wdd import PythonParserWeightedDeltaDebugger


def test_parser_wdd_only_sends_parseable_source_to_oracle_and_tests_empty_program():
    observed = []

    def oracle(source: str) -> bool:
        ast.parse(source)
        observed.append(source)
        return True

    result = PythonParserWeightedDeltaDebugger(oracle).reduce("value = 1\n")

    assert result.minimal_failing_input.strip() == ""
    assert any(not record.active_ids for record in result.test_records)
    assert observed and all(ast.parse(source) is not None for source in observed)


def test_parser_wdd_uses_source_token_weight_and_removes_irrelevant_statements():
    source = "alpha = 1\nraise ValueError('bug')\nvery_long_and_irrelevant_name = 3\n"
    result = PythonParserWeightedDeltaDebugger(
        lambda candidate: "raise ValueError" in candidate
    ).reduce(source)

    assert "raise ValueError" in result.minimal_failing_input
    assert "alpha" not in result.minimal_failing_input
    assert "irrelevant" not in result.minimal_failing_input
    assert all(record.weight > 0 for record in result.test_records)


def test_parser_wdd_preserves_comments_and_repairs_an_empty_required_suite():
    source = "# fmt: off\ndef target():\n    value = 1  # retain trivia\n"
    result = PythonParserWeightedDeltaDebugger(lambda candidate: "def target" in candidate).reduce(source)

    assert "# fmt: off" in result.minimal_failing_input
    assert "def target" in result.minimal_failing_input
    assert "pass" in result.minimal_failing_input
    ast.parse(result.minimal_failing_input)
