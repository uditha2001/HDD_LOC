import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bugsinpy_instrument_cli import instrument_source
from line_localization import LineCoverageRecord, LineClassification
from sbfl_score import rank_lines_with_predicate_context


def test_hybrid_predicate_context_blends_predicate_and_statement_scores():
    records = [
        LineCoverageRecord(test_id=1, failed=True, weight=1.0, lines=frozenset({2})),
        LineCoverageRecord(test_id=2, failed=True, weight=1.0, lines=frozenset({2})),
        LineCoverageRecord(test_id=3, failed=False, weight=1.0, lines=frozenset({3})),
    ]
    classification = {
        2: LineClassification(role="predicate", controlling_predicate=2),
        3: LineClassification(role="dependent", controlling_predicate=2),
    }

    ranking = rank_lines_with_predicate_context(records, classification, formula="ochiai")
    by_line = {line: score for line, score, _ in ranking}
    assert by_line[2] == 1.0
    assert by_line[3] == 1.0


def test_hybrid_predicate_context_uses_the_maximum_associated_predicate_score():
    records = [
        LineCoverageRecord(test_id=1, failed=True, weight=1.0, lines=frozenset({2, 4})),
        LineCoverageRecord(test_id=2, failed=False, weight=1.0, lines=frozenset({3, 4})),
        LineCoverageRecord(test_id=3, failed=True, weight=1.0, lines=frozenset({2})),
    ]
    classification = {
        2: LineClassification(role="predicate", controlling_predicate=2, controlling_predicates=(2,)),
        3: LineClassification(role="predicate", controlling_predicate=3, controlling_predicates=(3,)),
        4: LineClassification(role="dependent", controlling_predicate=3, controlling_predicates=(2, 3)),
    }

    ranking = rank_lines_with_predicate_context(records, classification, formula="ochiai", normalize=False)
    by_line = {line: score for line, score, _ in ranking}
    assert by_line[4] == 0.75


def test_hybrid_predicate_context_uses_dynamic_alpha_rules():
    records = [
        LineCoverageRecord(test_id=1, failed=True, weight=1.0, lines=frozenset({2, 3, 4})),
        LineCoverageRecord(test_id=2, failed=False, weight=1.0, lines=frozenset({2, 4})),
        LineCoverageRecord(test_id=3, failed=False, weight=1.0, lines=frozenset({3, 4})),
    ]
    classification = {
        2: LineClassification(role="predicate", controlling_predicate=2, controlling_predicates=(2,)),
        3: LineClassification(role="dependent", controlling_predicate=2, controlling_predicates=(2,)),
    }

    ranking = rank_lines_with_predicate_context(records, classification, formula="ochiai", normalize=False)
    by_line = {line: score for line, score, _ in ranking}
    assert by_line[2] == 0.7071067811865475
    assert by_line[3] == 0.7071067811865475
    assert by_line[4] == 0.5773502691896258


def test_instruments_if_and_while_and_records_original_source_lines(tmp_path, monkeypatch):
    source = """def classify(value):
    if value > 0:
        return 'positive'
    while value < 0:
        value += 1
    return 'zero'
"""
    result = instrument_source(source, "subject.py")

    assert [(site.predicate_id, site.line, site.kind) for site in result.predicates] == [
        (1, 2, "if"),
        (2, 4, "while"),
    ]
    assert result.source.count("\n") == source.count("\n")

    log_path = tmp_path / "pred_freq.txt"
    monkeypatch.setenv("HDD_LOC_PREDICATE_LOG", str(log_path))
    namespace = {}
    exec(compile(result.source, "subject.py", "exec"), namespace)
    assert namespace["classify"](1) == "positive"
    assert namespace["classify"](-1) == "zero"

    assert log_path.read_text(encoding="utf-8").splitlines() == [
        "LOG_PRED called: id=1 line=2 cond=1",
        "LOG_PRED called: id=1 line=2 cond=0",
        "LOG_PRED called: id=2 line=4 cond=1",
        "LOG_PRED called: id=2 line=4 cond=0",
    ]


def test_does_not_log_when_environment_variable_is_absent(monkeypatch):
    monkeypatch.delenv("HDD_LOC_PREDICATE_LOG", raising=False)
    result = instrument_source("if True:\n    value = 1\n")
    namespace = {}
    exec(compile(result.source, "subject.py", "exec"), namespace)
    assert namespace["value"] == 1
