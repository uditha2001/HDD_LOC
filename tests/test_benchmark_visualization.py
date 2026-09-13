from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_suite import METRIC_FIELDS, STATUS_FIELDS
from benchmark_visualization import _common_cases, generate_benchmark_graphs


def _write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_aggregate_graphs_compare_only_cases_shared_by_all_partitions():
    rows = [
        {"case": "black_1", "partition": "baseline_hdd"},
        {"case": "black_2", "partition": "baseline_hdd"},
        {"case": "black_1", "partition": "weighted_hdd"},
        {"case": "black_1", "partition": "ddmin_loc"},
    ]

    assert _common_cases(rows, ("baseline_hdd", "weighted_hdd", "ddmin_loc")) == {
        "black_1"
    }


def test_graphs_are_generated_from_reports_and_include_time_limited_data(tmp_path):
    metric_rows = []
    for case_index, case in enumerate(("black_1", "tqdm_1"), 1):
        for partition_index, partition in enumerate(
            ("baseline_hdd", "weighted_hdd", "ddmin_loc"), 1
        ):
            for budget in (5, "full"):
                metric_rows.append(
                    {
                        "case": case,
                        "project": case.split("_")[0],
                        "bug_id": case_index,
                        "input_kind": "tree",
                        "partition": partition,
                        "evidence_weighting": "unit",
                        "budget": budget,
                        "formula": "ochiai",
                        "method": "raw_sbfl",
                        "executions_available": 10,
                        "executions_in_budget": 5 if budget == 5 else 10,
                        "raw_evidence_count": 10,
                        "deduplicated_evidence_count": 8,
                        "ranked_lines": 100,
                        "faulty_line_covered": True,
                        "fault_rank": partition_index + case_index,
                        "tie_best_rank": partition_index,
                        "tie_worst_rank": partition_index + 2,
                        "tie_average_rank": partition_index + 1,
                        "exam_score": 0.1 * (partition_index + case_index),
                        "inspect_at_1": False,
                        "inspect_at_3": True,
                        "inspect_at_5": True,
                        "inspect_at_10": True,
                    }
                )
    status_rows = []
    for partition in ("baseline_hdd", "weighted_hdd", "ddmin_loc"):
        status_rows.append(
            {
                "case": "tqdm_1",
                "project": "tqdm",
                "bug_id": 1,
                "primary_module": "tqdm/contrib/__init__.py",
                "actual_fault_category": "semantic",
                "input_kind": "tree",
                "partition": partition,
                "status": "TIME_LIMIT" if partition == "ddmin_loc" else "COMPLETED",
                "message": "",
                "executions": 10,
                "fail": 3,
                "pass": 5,
                "invalid": 2,
                "minimal_input": "{}",
                "runtime_seconds": 1.0,
            }
        )
    _write_csv(tmp_path / "all_bugs_metrics.csv", METRIC_FIELDS, metric_rows)
    _write_csv(tmp_path / "all_bugs_status.csv", STATUS_FIELDS, status_rows)

    generated = generate_benchmark_graphs(tmp_path, formats=("png",))

    names = {path.name for path in generated}
    assert "exam_score_by_budget_raw_sbfl.png" in names
    assert "tie_average_rank_by_budget_raw_sbfl.png" in names
    assert "inspect_at_k_full_raw_sbfl.png" in names
    assert "exam_score_distribution_full_raw_sbfl.png" in names
    assert "exam_score_heatmap_full_ochiai_raw_sbfl.png" in names
    assert "candidate_outcome_balance.png" in names
    assert all(path.stat().st_size > 0 for path in generated)
