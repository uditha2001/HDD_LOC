"""Main integration script for HDD-LOC.

The pipeline is:
1. HDD minimizes the structured input and records pass/fail test cases.
2. line_localization replays each test case and records executed lines.
3. sbfl_score ranks the most suspicious lines.
"""

from __future__ import annotations

import csv
import math
import os
from importlib.util import module_from_spec, spec_from_file_location
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from hdd_baseline import HierarchicalDeltaDebugger as HDDBaselineDebugger
from hdd_algorithm import HDDResult, HierarchicalDeltaDebugger
from line_localization import collect_line_coverage, test_cases_from_hdd_result
from line_localization_baseline import collect_line_coverage as collect_line_coverage_baseline
from line_localization_baseline import test_cases_from_hdd_result as test_cases_from_hdd_result_baseline
from sbfl_baseline import rank_lines as rank_lines_baseline
from sbfl_score import rank_lines, render_report


FORMULA_SPECS: List[Tuple[str, str]] = [
    ("Ochiai", "ochiai"),
    ("Tarantula", "tarantula"),
    ("Jaccard", "jaccard"),
    ("DStar", "dstar2"),
]

KNOWN_FAULTY_LINES: Sequence[int] = (4,)


def _load_program(program_path: str):
    abs_path = os.path.abspath(program_path)
    spec = spec_from_file_location("_pipeline_buggy_program", abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load program at {program_path!r}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_oracle(program_path: str, reference_path: Optional[str] = None):
    """Build a Python differential oracle for the buggy program."""

    program_module = _load_program(program_path)
    reference_module = _load_program(reference_path or program_path.replace("_buggy", "_fixed"))

    def oracle(candidate: Any) -> bool:
        try:
            buggy_output = getattr(program_module, "run")(candidate)
            fixed_output = getattr(reference_module, "run")(candidate)
            return buggy_output != fixed_output
        except Exception:
            try:
                getattr(reference_module, "run")(candidate)
            except Exception:
                return True
            return True

    return oracle


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
    oracle = _build_oracle(program_path, reference_path=reference_path)

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

    oracle = _build_oracle(program_path, reference_path=reference_path)
    debugger = HierarchicalDeltaDebugger(oracle=oracle, weighting="subtree_size")
    hdd_result = debugger.reduce(structured_input)
    test_cases = test_cases_from_hdd_result(hdd_result, debugger)
    coverage = collect_line_coverage(program_path, test_cases, entry_function=entry_function)
    ranking = rank_lines(coverage, formula=formula, normalize=normalize)
    report = render_report(program_path, ranking, top_n=top_n)
    return hdd_result, ranking, report


if __name__ == "__main__":
    sample_input = [3, -1, 5, -2, 8]
    root = os.path.dirname(os.path.abspath(__file__))
    program_path = os.path.join(root, "sample_buggy_program.py")
    reference_path = os.path.join(root, "sample_fixed_program.py")
    comparison_rows: List[Dict[str, Any]] = []
    for formula_label, formula_name in FORMULA_SPECS:
        comparison_rows.extend(
            _run_weighted_and_baseline(
                structured_input=sample_input,
                program_path=program_path,
                reference_path=reference_path,
                entry_function="run",
                formula_label=formula_label,
                formula_name=formula_name,
                faulty_lines=KNOWN_FAULTY_LINES,
            )
        )

    outpath = os.path.join(root, "sbfl_comparison_metrics.csv")
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
    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)

    print(f"Wrote baseline-vs-weighted evaluation metrics to {outpath}")
