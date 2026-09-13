from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_suite import (
    DEFAULT_DDMIN_TIME_LIMIT_SECONDS,
    PARTITIONS,
    ensure_bugsinpy_container,
    ranking_metrics,
)
from execution_records import SourceLocation


def test_ranking_metrics_report_ties_and_exam_score():
    ranking = [
        SimpleNamespace(location=SourceLocation("code.py", 1), score=0.9),
        SimpleNamespace(location=SourceLocation("code.py", 2), score=0.8),
        SimpleNamespace(location=SourceLocation("code.py", 3), score=0.8),
        SimpleNamespace(location=SourceLocation("code.py", 4), score=0.1),
    ]

    metrics = ranking_metrics(ranking, {SourceLocation("code.py", 3)})

    assert metrics.deterministic_rank == 3
    assert metrics.tie_best_rank == 2
    assert metrics.tie_worst_rank == 3
    assert metrics.tie_average_rank == 2.5
    assert metrics.exam_score == 0.625
    assert metrics.inspect_at_1 is False
    assert metrics.inspect_at_3 is True


def test_all_comparison_partitions_are_enabled_by_default():
    assert PARTITIONS == ("baseline_hdd", "weighted_hdd", "ddmin_loc")
    assert DEFAULT_DDMIN_TIME_LIMIT_SECONDS == 900


def test_container_is_started_when_it_is_stopped(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[1] == "inspect":
            return SimpleNamespace(returncode=0, stdout="false\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="bugsinpy_bg\n", stderr="")

    monkeypatch.setattr("benchmark_suite.subprocess.run", fake_run)

    ensure_bugsinpy_container()

    assert calls == [
        ["docker", "inspect", "-f", "{{.State.Running}}", "bugsinpy_bg"],
        ["docker", "start", "bugsinpy_bg"],
    ]
