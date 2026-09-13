"""Main integration script for HDD-LOC.

This is the only module that should need bug-specific wiring.
It loads the buggy/fixed programs, builds the oracle, runs weighted and
baseline HDD, replays coverage, and writes the SBFL comparison results.
"""

from __future__ import annotations

import csv
import argparse
import json
import os
from importlib.util import module_from_spec, spec_from_file_location
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from black2_adapter import BUGGY_ROOT as BLACK2_BUGGY_ROOT
from black2_adapter import FIXED_ROOT as BLACK2_FIXED_ROOT
from black2_adapter import Black2Adapter
from hdd_baseline import HierarchicalDeltaDebugger as HDDBaselineDebugger
from hdd_algorithm import HDDResult, HierarchicalDeltaDebugger
from line_localization import LineCoverageRecord as WeightedLineCoverageRecord
from line_localization import collect_line_coverage, test_cases_from_hdd_result
from line_localization_baseline import collect_line_coverage as collect_line_coverage_baseline
from line_localization_baseline import LineCoverageRecord as BaselineLineCoverageRecord
from line_localization_baseline import test_cases_from_hdd_result as test_cases_from_hdd_result_baseline
from sbfl_baseline import rank_lines as rank_lines_baseline
from sbfl_score import rank_lines, render_report
from execution_records import (
    CandidateEvaluation,
    ExecutionRecord,
    Outcome,
    RecordingOracle,
    SourceLocation,
    build_execution_records,
    write_execution_records_jsonl,
)
from local_adapter import LocalDifferentialAdapter


FORMULA_SPECS: List[Tuple[str, str]] = [
    ("Ochiai", "ochiai"),
    ("Tarantula", "tarantula"),
    ("Jaccard", "jaccard"),
    ("DStar", "dstar2"),
]

KNOWN_FAULTY_LINES: Sequence[int] = ()


def _load_program(program_path: str):
    abs_path = os.path.abspath(program_path)
    spec = spec_from_file_location("_pipeline_buggy_program", abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load program at {program_path!r}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_differential_oracle(
    program_path: str,
    reference_path: Optional[str] = None,
    entry_function: str = "run",
):
    """Build a differential oracle that compares buggy and fixed runs."""

    program_module = _load_program(program_path)
    reference_module = _load_program(reference_path or program_path.replace("_buggy", "_fixed"))
    buggy_entry = getattr(program_module, entry_function)
    reference_entry = getattr(reference_module, entry_function)

    if not callable(buggy_entry):
        raise AttributeError(f"{program_path!r} has no callable {entry_function!r}")
    if not callable(reference_entry):
        raise AttributeError(f"{reference_module.__file__!r} has no callable {entry_function!r}")

    def oracle(candidate: Any) -> bool:
        try:
            buggy_output = buggy_entry(candidate)
            fixed_output = reference_entry(candidate)
            return buggy_output != fixed_output
        except Exception:
            try:
                reference_entry(candidate)
            except Exception:
                return True
            return True

    return oracle


def _candidate_to_source(candidate: Any) -> str:
    if isinstance(candidate, str):
        return candidate
    if isinstance(candidate, (list, tuple)):
        return "".join(str(part) for part in candidate)
    return str(candidate)


def _black2_source_to_structured_input(source: str) -> List[str]:
    return source.splitlines(keepends=True)


def build_black2_oracle(
    buggy_root: Optional[str] = None,
    fixed_root: Optional[str] = None,
    timeout: int = 30,
    collect_coverage: bool = True,
) -> Tuple[Callable[[Any], bool], Black2Adapter]:
    adapter = Black2Adapter(
        buggy_root=buggy_root or BLACK2_BUGGY_ROOT,
        fixed_root=fixed_root or BLACK2_FIXED_ROOT,
        timeout=timeout,
        collect_coverage=collect_coverage,
    )

    def oracle(candidate: Any) -> bool:
        return adapter.oracle(_candidate_to_source(candidate))

    return oracle, adapter


def collect_local_hdd_execution_records(
    structured_input: Any,
    program_path: str,
    reference_path: str,
    entry_function: str = "run",
    weighting: str = "subtree_size",
    records_output_path: Optional[str] = None,
) -> Tuple[HDDResult, Tuple[ExecutionRecord, ...]]:
    """Run HDD once while collecting canonical local execution records."""

    adapter = LocalDifferentialAdapter(
        buggy_program=program_path,
        fixed_program=reference_path,
        entry_function=entry_function,
    )
    oracle = RecordingOracle(adapter.evaluate)
    debugger = HierarchicalDeltaDebugger(oracle=oracle, weighting=weighting)
    result = debugger.reduce(structured_input)
    records = build_execution_records(result, oracle.observations)
    if records_output_path is not None:
        write_execution_records_jsonl(records, records_output_path)
    return result, records


def collect_black2_hdd_execution_records(
    structured_input: Any,
    buggy_root: Optional[str] = None,
    fixed_root: Optional[str] = None,
    timeout: int = 30,
    weighting: str = "subtree_size",
    records_output_path: Optional[str] = None,
) -> Tuple[HDDResult, Tuple[ExecutionRecord, ...]]:
    """Run Black HDD once; oracle outcome and coverage share one execution."""

    adapter = Black2Adapter(
        buggy_root=buggy_root or BLACK2_BUGGY_ROOT,
        fixed_root=fixed_root or BLACK2_FIXED_ROOT,
        timeout=timeout,
        collect_coverage=True,
    )

    def evaluator(candidate: Any) -> CandidateEvaluation:
        result = adapter.evaluate(_candidate_to_source(candidate))
        return CandidateEvaluation(
            outcome=Outcome(result.outcome),
            coverage=frozenset(
                SourceLocation(filename, line)
                for filename, line in result.coverage
            ),
            structurally_valid=result.structurally_valid,
            semantically_valid=result.semantically_valid,
            runtime_ms=result.runtime_ms,
        )

    oracle = RecordingOracle(evaluator, serializer=_candidate_to_source)
    debugger = HierarchicalDeltaDebugger(oracle=oracle, weighting=weighting)
    result = debugger.reduce(structured_input)
    records = build_execution_records(result, oracle.observations)
    if records_output_path is not None:
        write_execution_records_jsonl(records, records_output_path)
    return result, records


def _evaluate_black2_test_cases(
    adapter: Black2Adapter,
    test_cases: Iterable[Any],
    weighted: bool,
) -> List[Any]:
    records: List[Any] = []
    for case in test_cases:
        result = adapter.evaluate(_candidate_to_source(case.candidate))
        lines = frozenset(line for _, line in result.coverage)
        if weighted:
            records.append(
                WeightedLineCoverageRecord(
                    test_id=case.test_id,
                    failed=result.outcome == "FAIL",
                    weight=case.weight,
                    lines=lines,
                )
            )
        else:
            records.append(
                BaselineLineCoverageRecord(
                    test_id=case.test_id,
                    failed=result.outcome == "FAIL",
                    lines=lines,
                )
            )
    return records


def _first_fault_rank(ranking: List[Tuple[int, float, Any]], faulty_lines: Sequence[int]) -> int:
    for index, (line_no, _, _) in enumerate(ranking, start=1):
        if line_no in faulty_lines:
            return index
    return len(ranking) + 1 if ranking else 1


def _top_n_hit(ranking: List[Tuple[int, float, Any]], faulty_lines: Sequence[int], top_n: int) -> bool:
    ranked_lines = {line_no for line_no, _, _ in ranking[:top_n]}
    return any(line in ranked_lines for line in faulty_lines)


def _exam_score(first_rank: int, total_lines: int) -> float:
    if total_lines <= 0:
        return 0.0
    return first_rank / float(total_lines)


def _compute_metrics(ranking: List[Tuple[int, float, Any]], faulty_lines: Sequence[int]) -> Dict[str, float]:
    total_lines = len(ranking)
    first_rank = _first_fault_rank(ranking, faulty_lines)
    return {
        "Exam_Score": _exam_score(first_rank, total_lines),
        "First_Fault_Rank": float(first_rank),
        "Top_1": 1.0 if _top_n_hit(ranking, faulty_lines, 1) else 0.0,
        "Top_3": 1.0 if _top_n_hit(ranking, faulty_lines, 3) else 0.0,
        "Top_5": 1.0 if _top_n_hit(ranking, faulty_lines, 5) else 0.0,
        "Top_10": 1.0 if _top_n_hit(ranking, faulty_lines, 10) else 0.0,
    }


def _run_weighted_and_baseline(
    structured_input: Any,
    program_path: str,
    reference_path: Optional[str],
    entry_function: str,
    formula_label: str,
    formula_name: str,
    faulty_lines: Sequence[int],
) -> List[Dict[str, Any]]:
    oracle = build_differential_oracle(program_path, reference_path=reference_path, entry_function=entry_function)

    weighted_debugger = HierarchicalDeltaDebugger(oracle=oracle, weighting="subtree_size")
    weighted_result = weighted_debugger.reduce(structured_input)
    weighted_cases = test_cases_from_hdd_result(weighted_result, weighted_debugger)
    weighted_coverage = collect_line_coverage(program_path, weighted_cases, entry_function=entry_function)
    weighted_ranking = rank_lines(weighted_coverage, formula=formula_name, normalize=True)
    weighted_metrics = _compute_metrics(weighted_ranking, faulty_lines)

    baseline_debugger = HDDBaselineDebugger(oracle=oracle)
    baseline_result = baseline_debugger.reduce(structured_input)
    baseline_cases = test_cases_from_hdd_result_baseline(baseline_result, baseline_debugger)
    baseline_coverage = collect_line_coverage_baseline(program_path, baseline_cases, entry_function=entry_function)
    baseline_ranking = rank_lines_baseline(baseline_coverage, formula=formula_name, normalize=True)
    baseline_metrics = _compute_metrics(baseline_ranking, faulty_lines)

    rows: List[Dict[str, Any]] = []
    for method_name, metrics in (("Weighted", weighted_metrics), ("Baseline", baseline_metrics)):
        row = {
            "Program": os.path.basename(program_path),
            "Formula": formula_label,
            "Method": method_name,
            "Faulty_Lines": ";".join(str(line) for line in faulty_lines),
            "Total_Lines": float(len(weighted_ranking)),
        }
        row.update(metrics)
        rows.append(row)

    return rows


def _run_black2_weighted_and_baseline(
    structured_input: Any,
    seed_path: str,
    buggy_root: Optional[str],
    fixed_root: Optional[str],
    timeout: int,
    collect_coverage: bool,
    formula_label: str,
    formula_name: str,
    faulty_lines: Sequence[int],
) -> List[Dict[str, Any]]:
    oracle, adapter = build_black2_oracle(
        buggy_root=buggy_root,
        fixed_root=fixed_root,
        timeout=timeout,
        collect_coverage=collect_coverage,
    )

    weighted_debugger = HierarchicalDeltaDebugger(oracle=oracle, weighting="subtree_size")
    weighted_result = weighted_debugger.reduce(structured_input)
    weighted_cases = test_cases_from_hdd_result(weighted_result, weighted_debugger)
    weighted_coverage = _evaluate_black2_test_cases(adapter, weighted_cases, weighted=True)
    weighted_ranking = rank_lines(weighted_coverage, formula=formula_name, normalize=True)
    weighted_metrics = _compute_metrics(weighted_ranking, faulty_lines)

    baseline_debugger = HDDBaselineDebugger(oracle=oracle)
    baseline_result = baseline_debugger.reduce(structured_input)
    baseline_cases = test_cases_from_hdd_result_baseline(baseline_result, baseline_debugger)
    baseline_coverage = _evaluate_black2_test_cases(adapter, baseline_cases, weighted=False)
    baseline_ranking = rank_lines_baseline(baseline_coverage, formula=formula_name, normalize=True)
    baseline_metrics = _compute_metrics(baseline_ranking, faulty_lines)

    rows: List[Dict[str, Any]] = []
    for method_name, metrics in (("Weighted", weighted_metrics), ("Baseline", baseline_metrics)):
        row = {
            "Program": os.path.basename(seed_path),
            "Formula": formula_label,
            "Method": method_name,
            "Faulty_Lines": ";".join(str(line) for line in faulty_lines),
            "Total_Lines": float(len(weighted_ranking)),
        }
        row.update(metrics)
        rows.append(row)

    return rows


def run_pipeline(
    structured_input: Any,
    program_path: str,
    reference_path: Optional[str] = None,
    entry_function: str = "run",
    formula: str = "ochiai",
    normalize: bool = True,
    top_n: Optional[int] = None,
) -> Tuple[HDDResult, list, str]:
    """Run the full HDD-LOC pipeline for the Python buggy/fixed comparison demo."""

    oracle = build_differential_oracle(program_path, reference_path=reference_path, entry_function=entry_function)
    debugger = HierarchicalDeltaDebugger(oracle=oracle, weighting="subtree_size")
    hdd_result = debugger.reduce(structured_input)
    test_cases = test_cases_from_hdd_result(hdd_result, debugger)
    coverage = collect_line_coverage(program_path, test_cases, entry_function=entry_function)
    ranking = rank_lines(coverage, formula=formula, normalize=normalize)
    report = render_report(program_path, ranking, top_n=top_n)
    return hdd_result, ranking, report


def _load_structured_input(input_path: Optional[str], input_json: Optional[str]) -> Any:
    if input_path:
        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)
    if input_json is not None:
        return json.loads(input_json)
    return [3, -1, 5, -2, 8]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the HDD-LOC pipeline on a buggy/fixed program pair.")
    parser.add_argument(
        "--cases",
        default=None,
        help="Comma-separated BugsInPy cases; default with no legacy inputs is every selected case.",
    )
    parser.add_argument(
        "--results-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results"),
        help="Directory for all-bugs execution records and analytical reports.",
    )
    parser.add_argument(
        "--partitions",
        default=None,
        help=(
            "Comma-separated minimizers: baseline_hdd, weighted_hdd, ddmin_loc; "
            "default is all three."
        ),
    )
    parser.add_argument("--case-timeout", type=int, default=30)
    parser.add_argument(
        "--ddmin-time-limit",
        type=float,
        default=15 * 60,
        help=(
            "DDMin-LOC wall-clock limit per bug in seconds; default 900 "
            "matches the paper's 15-minute evaluation budget."
        ),
    )
    parser.add_argument(
        "--reuse-execution-records",
        action="store_true",
        help="Regenerate reports from existing JSONL records without rerunning HDD.",
    )
    parser.add_argument("--buggy-program", default=None, help="Path to the buggy program for the local Python pipeline.")
    parser.add_argument("--fixed-program", default=None, help="Path to the fixed program for the local Python pipeline.")
    parser.add_argument("--black2-seed-file", default=None, help="Path to a Black2 source file to reduce inside BugsInPy.")
    parser.add_argument("--black2-buggy-root", default=None, help="Host path to the Black2 buggy BugsInPy checkout.")
    parser.add_argument("--black2-fixed-root", default=None, help="Host path to the Black2 fixed BugsInPy checkout.")
    parser.add_argument("--black2-timeout", type=int, default=30, help="Timeout in seconds for each Black2 container run.")
    parser.add_argument("--input-json", default=None, help="Structured input as a JSON string.")
    parser.add_argument("--input-file", default=None, help="Path to a JSON file containing the structured input.")
    parser.add_argument("--entry-function", default="run")
    parser.add_argument("--faulty-lines", default="", help="Comma-separated faulty line numbers for metric reporting.")
    parser.add_argument("--output-csv", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sbfl_comparison_metrics.csv"))
    args = parser.parse_args(list(argv) if argv is not None else None)

    faulty_lines = tuple(int(part) for part in args.faulty_lines.split(",") if part.strip())
    comparison_rows: List[Dict[str, Any]] = []
    if args.black2_seed_file:
        with open(args.black2_seed_file, "r", encoding="utf-8") as f:
            structured_input = _black2_source_to_structured_input(f.read())

        for formula_label, formula_name in FORMULA_SPECS:
            comparison_rows.extend(
                _run_black2_weighted_and_baseline(
                    structured_input=structured_input,
                    seed_path=args.black2_seed_file,
                    buggy_root=args.black2_buggy_root,
                    fixed_root=args.black2_fixed_root,
                    timeout=args.black2_timeout,
                    collect_coverage=True,
                    formula_label=formula_label,
                    formula_name=formula_name,
                    faulty_lines=faulty_lines,
                )
            )
    elif args.buggy_program or args.fixed_program:
        if not args.buggy_program or not args.fixed_program:
            raise SystemExit("Provide both --buggy-program and --fixed-program.")
        structured_input = _load_structured_input(args.input_file, args.input_json)
        for formula_label, formula_name in FORMULA_SPECS:
            comparison_rows.extend(
                _run_weighted_and_baseline(
                    structured_input=structured_input,
                    program_path=args.buggy_program,
                    reference_path=args.fixed_program,
                    entry_function=args.entry_function,
                    formula_label=formula_label,
                    formula_name=formula_name,
                    faulty_lines=faulty_lines,
                )
            )
    else:
        from benchmark_suite import run_selected_benchmarks

        case_keys = None
        if args.cases:
            case_keys = tuple(part.strip() for part in args.cases.split(",") if part.strip())
        partitions = None
        if args.partitions:
            partitions = tuple(
                part.strip() for part in args.partitions.split(",") if part.strip()
            )
        output = run_selected_benchmarks(
            case_keys=case_keys,
            partitions=partitions,
            results_dir=args.results_dir,
            timeout=args.case_timeout,
            ddmin_time_limit_seconds=args.ddmin_time_limit,
            reuse_execution_records=args.reuse_execution_records,
        )
        print(f"Wrote all-bugs HDD-LOC reports to {output}")
        return 0

    fieldnames = [
        "Program",
        "Formula",
        "Method",
        "Faulty_Lines",
        "Total_Lines",
        "Exam_Score",
        "First_Fault_Rank",
        "Top_1",
        "Top_3",
        "Top_5",
        "Top_10",
    ]
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)

    print(f"Wrote baseline-vs-weighted evaluation metrics to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
