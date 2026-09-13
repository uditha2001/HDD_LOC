"""Run the selected BugsInPy cases and write raw HDD-LOC reports."""

from __future__ import annotations

import csv
import json
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from benchmark_cases import CASES_BY_KEY, SELECTED_CASES, BenchmarkCase
from benchmark_visualization import generate_benchmark_graphs
from ddmin_loc_adapter import run_original_ddmin_loc
from execution_records import (
    ExecutionRecord,
    Outcome,
    RecordingOracle,
    SourceLocation,
    build_execution_records,
    read_execution_records_jsonl,
    write_execution_records_jsonl,
)
from ground_truth import deleted_buggy_lines, executable_ground_truth
from hdd_algorithm import HierarchicalDeltaDebugger as WeightedHDD
from hdd_baseline import HierarchicalDeltaDebugger as BaselineHDD
from passive_pipeline import DEFAULT_EXECUTION_BUDGETS, PassiveAnalysis, analyze_execution_records


PARTITIONS = ("baseline_hdd", "weighted_hdd", "ddmin_loc")
DEFAULT_DDMIN_TIME_LIMIT_SECONDS = 15 * 60


@dataclass(frozen=True)
class RankingMetrics:
    faulty_line_covered: bool
    deterministic_rank: int
    tie_best_rank: int
    tie_worst_rank: int
    tie_average_rank: float
    exam_score: float
    inspect_at_1: bool
    inspect_at_3: bool
    inspect_at_5: bool
    inspect_at_10: bool


def ensure_bugsinpy_container(container_name: str = "bugsinpy_bg") -> None:
    """Make the no-argument suite self-contained with respect to Docker state."""

    try:
        inspected = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"cannot inspect BugsInPy Docker container: {exc}") from exc
    if inspected.returncode != 0:
        raise RuntimeError(
            f"BugsInPy Docker container {container_name!r} is unavailable: "
            f"{inspected.stderr.strip()}"
        )
    if inspected.stdout.strip().lower() == "true":
        return
    started = subprocess.run(
        ["docker", "start", container_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    if started.returncode != 0:
        raise RuntimeError(
            f"could not start BugsInPy Docker container {container_name!r}: "
            f"{started.stderr.strip()}"
        )


def _score(item: Any) -> float:
    return float(item.score)


def ranking_metrics(
    ranking: Sequence[Any],
    faulty_locations: Iterable[SourceLocation],
) -> RankingMetrics:
    faults = set(faulty_locations)
    positions = [index for index, item in enumerate(ranking, 1) if item.location in faults]
    total = len(ranking)
    if not positions:
        missing_rank = total + 1
        return RankingMetrics(
            False,
            missing_rank,
            missing_rank,
            missing_rank,
            float(missing_rank),
            1.0,
            False,
            False,
            False,
            False,
        )

    deterministic = min(positions)
    fault_score = max(_score(ranking[position - 1]) for position in positions)
    tied_positions = [
        index for index, item in enumerate(ranking, 1) if _score(item) == fault_score
    ]
    best = min(tied_positions)
    worst = max(tied_positions)
    average = (best + worst) / 2.0
    exam = average / total if total else 1.0
    return RankingMetrics(
        True,
        deterministic,
        best,
        worst,
        average,
        exam,
        average <= 1,
        average <= 3,
        average <= 5,
        average <= 10,
    )


def _collect_records(
    case: BenchmarkCase,
    partition: str,
    timeout: int,
    ddmin_time_limit_seconds: float,
):
    runtime = case.build_runtime(timeout=timeout)
    if partition == "ddmin_loc":
        return run_original_ddmin_loc(
            runtime.serializer(runtime.structured_input),
            runtime.serialized_evaluator,
            time_limit_seconds=ddmin_time_limit_seconds,
        )

    oracle = RecordingOracle(runtime.evaluator, serializer=runtime.serializer)
    if partition == "baseline_hdd":
        debugger = BaselineHDD(oracle=oracle)
    elif partition == "weighted_hdd":
        debugger = WeightedHDD(oracle=oracle, weighting="subtree_size")
    else:
        raise ValueError(f"unknown HDD partition: {partition}")
    result = debugger.reduce(runtime.structured_input)
    records = build_execution_records(result, oracle.observations)
    return result, records


def _method_rankings(analysis: PassiveAnalysis):
    return (
        ("raw_sbfl", analysis.raw_ranking),
        ("deduplicated_raw_sbfl", analysis.deduplicated_raw_ranking),
    )


METRIC_FIELDS = [
    "case",
    "project",
    "bug_id",
    "input_kind",
    "partition",
    "evidence_weighting",
    "budget",
    "formula",
    "method",
    "executions_available",
    "executions_in_budget",
    "raw_evidence_count",
    "deduplicated_evidence_count",
    "ranked_lines",
    "faulty_line_covered",
    "fault_rank",
    "tie_best_rank",
    "tie_worst_rank",
    "tie_average_rank",
    "exam_score",
    "inspect_at_1",
    "inspect_at_3",
    "inspect_at_5",
    "inspect_at_10",
]


RANKING_FIELDS = [
    "case",
    "partition",
    "budget",
    "formula",
    "method",
    "rank",
    "filename",
    "line",
    "is_faulty",
    "score",
    "ef",
    "ep",
    "nf",
    "np",
]


STATUS_FIELDS = [
    "case",
    "project",
    "bug_id",
    "primary_module",
    "actual_fault_category",
    "input_kind",
    "partition",
    "status",
    "message",
    "executions",
    "fail",
    "pass",
    "invalid",
    "minimal_input",
    "runtime_seconds",
]


def _write_aggregate_report(metric_path: Path, output_path: Path) -> None:
    with metric_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    groups: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            row["partition"],
            row["evidence_weighting"],
            row["budget"],
            row["formula"],
            row["method"],
        )
        groups.setdefault(key, []).append(row)
    fields = [
        "partition",
        "evidence_weighting",
        "budget",
        "formula",
        "method",
        "cases",
        "mean_exam_score",
        "median_exam_score",
        "fault_coverage_rate",
        "inspect_at_1_rate",
        "inspect_at_3_rate",
        "inspect_at_5_rate",
        "inspect_at_10_rate",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key in sorted(groups):
            members = groups[key]
            exams = [float(row["exam_score"]) for row in members]

            def rate(field: str) -> float:
                return sum(row[field] == "True" for row in members) / len(members)

            writer.writerow(
                {
                    "partition": key[0],
                    "evidence_weighting": key[1],
                    "budget": key[2],
                    "formula": key[3],
                    "method": key[4],
                    "cases": len(members),
                    "mean_exam_score": statistics.fmean(exams),
                    "median_exam_score": statistics.median(exams),
                    "fault_coverage_rate": rate("faulty_line_covered"),
                    "inspect_at_1_rate": rate("inspect_at_1"),
                    "inspect_at_3_rate": rate("inspect_at_3"),
                    "inspect_at_5_rate": rate("inspect_at_5"),
                    "inspect_at_10_rate": rate("inspect_at_10"),
                }
            )


def run_selected_benchmarks(
    *,
    case_keys: Optional[Sequence[str]] = None,
    partitions: Optional[Sequence[str]] = None,
    results_dir: str | Path | None = None,
    budgets: Sequence[Optional[int]] = DEFAULT_EXECUTION_BUDGETS,
    formulas: Sequence[str] = ("ochiai", "jaccard", "tarantula", "dstar2"),
    timeout: int = 30,
    ddmin_time_limit_seconds: float = DEFAULT_DDMIN_TIME_LIMIT_SECONDS,
    ranking_limit: int = 100,
    reuse_execution_records: bool = False,
) -> Path:
    """Execute all configured cases and preserve records plus CSV reports."""

    if case_keys is None:
        cases = SELECTED_CASES
    else:
        unknown = sorted(set(case_keys).difference(CASES_BY_KEY))
        if unknown:
            raise ValueError(f"unknown benchmark case(s): {', '.join(unknown)}")
        cases = tuple(CASES_BY_KEY[key] for key in case_keys)

    selected_partitions = PARTITIONS if partitions is None else tuple(partitions)
    unknown_partitions = sorted(set(selected_partitions).difference(PARTITIONS))
    if unknown_partitions:
        raise ValueError(
            f"unknown benchmark partition(s): {', '.join(unknown_partitions)}"
        )
    if not selected_partitions:
        raise ValueError("at least one benchmark partition is required")
    if ddmin_time_limit_seconds <= 0:
        raise ValueError("DDMin-LOC time limit must be positive")

    if not reuse_execution_records:
        ensure_bugsinpy_container()

    if results_dir is None:
        results_dir = Path(__file__).resolve().parent / "benchmark_results"
    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    records_dir = output / "execution_records"
    records_dir.mkdir(parents=True, exist_ok=True)

    metric_path = output / "all_bugs_metrics.csv"
    ranking_path = output / "all_bugs_rankings.csv"
    status_path = output / "all_bugs_status.csv"
    config_path = output / "run_config.json"
    config_path.write_text(
        json.dumps(
            {
                "cases": [case.key for case in cases],
                "partitions": list(selected_partitions),
                "budgets": ["full" if budget is None else budget for budget in budgets],
                "formulas": list(formulas),
                "ddmin_time_limit_seconds": ddmin_time_limit_seconds,
                "ground_truth_usage": "evaluation only after ranking",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with (
        metric_path.open("w", newline="", encoding="utf-8") as metric_stream,
        ranking_path.open("w", newline="", encoding="utf-8") as ranking_stream,
        status_path.open("w", newline="", encoding="utf-8") as status_stream,
    ):
        metric_writer = csv.DictWriter(metric_stream, fieldnames=METRIC_FIELDS)
        ranking_writer = csv.DictWriter(ranking_stream, fieldnames=RANKING_FIELDS)
        status_writer = csv.DictWriter(status_stream, fieldnames=STATUS_FIELDS)
        metric_writer.writeheader()
        ranking_writer.writeheader()
        status_writer.writeheader()

        for case in cases:
            for partition in selected_partitions:
                started = time.perf_counter()
                try:
                    record_path = records_dir / f"{case.key}__{partition}.jsonl"
                    if reuse_execution_records and record_path.exists():
                        result = None
                        records = read_execution_records_jsonl(record_path)
                    else:
                        result, records = _collect_records(
                            case,
                            partition,
                            timeout,
                            ddmin_time_limit_seconds,
                        )
                    if not records or records[0].outcome is not Outcome.FAIL:
                        first = records[0].outcome.value if records else "NO_EXECUTION"
                        raise RuntimeError(f"seed did not reproduce semantic bug: {first}")
                    write_execution_records_jsonl(
                        records, record_path
                    )
                    weighting = (
                        "inverse_hdd" if partition == "weighted_hdd" else "unit"
                    )
                    analyses = analyze_execution_records(
                        records,
                        formulas=formulas,
                        budgets=budgets,
                        evidence_weighting=weighting,
                    )

                    # Ground truth is intentionally loaded only after all
                    # candidate generation and ranking have completed.
                    faulty_locations = executable_ground_truth(
                        deleted_buggy_lines(case.patch_path), case.buggy_root
                    )
                    for analysis in analyses:
                        budget_label = "full" if analysis.budget is None else analysis.budget
                        for method, ranking in _method_rankings(analysis):
                            metrics = ranking_metrics(ranking, faulty_locations)
                            metric_writer.writerow(
                                {
                                    "case": case.key,
                                    "project": case.project,
                                    "bug_id": case.bug_id,
                                    "input_kind": case.input_kind,
                                    "partition": partition,
                                    "evidence_weighting": weighting,
                                    "budget": budget_label,
                                    "formula": analysis.formula,
                                    "method": method,
                                    "executions_available": len(records),
                                    "executions_in_budget": analysis.original_execution_count,
                                    "raw_evidence_count": analysis.raw_evidence_count,
                                    "deduplicated_evidence_count": analysis.deduplicated_evidence_count,
                                    "ranked_lines": len(ranking),
                                    "faulty_line_covered": metrics.faulty_line_covered,
                                    "fault_rank": metrics.deterministic_rank,
                                    "tie_best_rank": metrics.tie_best_rank,
                                    "tie_worst_rank": metrics.tie_worst_rank,
                                    "tie_average_rank": metrics.tie_average_rank,
                                    "exam_score": metrics.exam_score,
                                    "inspect_at_1": metrics.inspect_at_1,
                                    "inspect_at_3": metrics.inspect_at_3,
                                    "inspect_at_5": metrics.inspect_at_5,
                                    "inspect_at_10": metrics.inspect_at_10,
                                }
                            )
                            selected_ranks = set(range(1, min(ranking_limit, len(ranking)) + 1))
                            selected_ranks.update(
                                index for index, item in enumerate(ranking, 1)
                                if item.location in faulty_locations
                            )
                            for index in sorted(selected_ranks):
                                item = ranking[index - 1]
                                observed = item.spectrum
                                ranking_writer.writerow(
                                    {
                                        "case": case.key,
                                        "partition": partition,
                                        "budget": budget_label,
                                        "formula": analysis.formula,
                                        "method": method,
                                        "rank": index,
                                        "filename": item.location.filename,
                                        "line": item.location.line,
                                        "is_faulty": item.location in faulty_locations,
                                        "score": _score(item),
                                        "ef": observed.ef,
                                        "ep": observed.ep,
                                        "nf": observed.nf,
                                        "np": observed.np,
                                    }
                                )

                    counts = {outcome: 0 for outcome in Outcome}
                    for record in records:
                        counts[record.outcome] += 1
                    timed_out = bool(getattr(result, "timed_out", False))
                    status_writer.writerow(
                        {
                            "case": case.key,
                            "project": case.project,
                            "bug_id": case.bug_id,
                            "primary_module": case.primary_module,
                            "actual_fault_category": case.fault_category,
                            "input_kind": case.input_kind,
                            "partition": partition,
                            "status": "TIME_LIMIT" if timed_out else "COMPLETED",
                            "message": (
                                "DDMin-LOC reached its "
                                f"{ddmin_time_limit_seconds:g}-second wall-clock limit; "
                                "metrics use all completed candidate executions"
                                if timed_out
                                else ""
                            ),
                            "executions": len(records),
                            "fail": counts[Outcome.FAIL],
                            "pass": counts[Outcome.PASS],
                            "invalid": counts[Outcome.INVALID],
                            "minimal_input": (
                                repr(result.minimal_failing_input)
                                if result is not None else "reused execution records"
                            ),
                            "runtime_seconds": time.perf_counter() - started,
                        }
                    )
                    metric_stream.flush()
                    ranking_stream.flush()
                    status_stream.flush()
                    suffix = " before time limit" if timed_out else " completed"
                    print(f"[{case.key}] {partition}: {len(records)} executions{suffix}")
                except Exception as exc:
                    status_writer.writerow(
                        {
                            "case": case.key,
                            "project": case.project,
                            "bug_id": case.bug_id,
                            "primary_module": case.primary_module,
                            "actual_fault_category": case.fault_category,
                            "input_kind": case.input_kind,
                            "partition": partition,
                            "status": "ERROR",
                            "message": f"{type(exc).__name__}: {exc}",
                            "executions": 0,
                            "fail": 0,
                            "pass": 0,
                            "invalid": 0,
                            "minimal_input": "",
                            "runtime_seconds": time.perf_counter() - started,
                        }
                    )
                    status_stream.flush()
                    print(f"[{case.key}] {partition}: ERROR: {exc}")

    _write_aggregate_report(metric_path, output / "aggregate_metrics.csv")
    generated_graphs = generate_benchmark_graphs(output)
    print(f"Generated {len(generated_graphs)} graph files in {output / 'graphs'}")
    return output
