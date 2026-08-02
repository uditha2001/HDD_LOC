"""Unweighted SBFL baseline: count-based suspiciousness scoring.

This module provides the same SBFL formulas as the weighted implementation
but computes ef/ep/nf/np using simple test counts (no weights).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from line_localization_baseline import LineCoverageRecord


@dataclass(frozen=True)
class LineSpectrum:
    line: int
    ef: float
    ep: float
    nf: float
    np: float


@dataclass(frozen=True)
class LineSuspiciousness:
    line: int
    spectrum: LineSpectrum
    score: float


def aggregate_line_spectrum(records: List[LineCoverageRecord]) -> Dict[int, LineSpectrum]:
    all_lines = {line for record in records for line in record.lines}
    total_failed = sum(1 for record in records if record.failed)
    total_passed = sum(1 for record in records if not record.failed)

    spectrum: Dict[int, LineSpectrum] = {}
    for line in all_lines:
        ef = sum(1 for record in records if record.failed and line in record.lines)
        ep = sum(1 for record in records if not record.failed and line in record.lines)
        nf = total_failed - ef
        np_ = total_passed - ep
        spectrum[line] = LineSpectrum(line=line, ef=ef, ep=ep, nf=nf, np=np_)
    return spectrum


def _sbfl_score(ef: float, ep: float, nf: float, np_: float, formula: str) -> float:
    formula = formula.lower()
    if formula == "tarantula":
        fail_ratio = ef / (ef + nf) if (ef + nf) else 0.0
        pass_ratio = ep / (ep + np_) if (ep + np_) else 0.0
        denom = fail_ratio + pass_ratio
        return fail_ratio / denom if denom else 0.0
    if formula == "ochiai":
        denom = math.sqrt((ef + nf) * (ef + ep))
        return ef / denom if denom else 0.0
    if formula == "jaccard":
        denom = ef + nf + ep
        return ef / denom if denom else 0.0
    if formula == "dstar2":
        denom = ep + nf
        if denom == 0:
            return float("inf") if ef > 0 else 0.0
        return (ef * ef) / denom
    raise ValueError(f"unknown SBFL formula: {formula!r}")


def rank_lines(records: List[LineCoverageRecord], formula: str = "ochiai", normalize: bool = True) -> List[Tuple[int, float, LineSpectrum]]:
    spectrum = aggregate_line_spectrum(records)
    scored = [(line, _sbfl_score(s.ef, s.ep, s.nf, s.np, formula), s) for line, s in spectrum.items()]
    if normalize and scored:
        finite_scores = [score for _, score, _ in scored if math.isfinite(score)]
        max_score = max(finite_scores) if finite_scores else 1.0
        max_score = max_score or 1.0
        scored = [
            (line, score if not math.isfinite(score) else score / max_score, spec)
            for line, score, spec in scored
        ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def rank_lines_from_hdd_result(result: Any, debugger: Any, program_path: str, entry_function: str = "run", formula: str = "ochiai", normalize: bool = True) -> List[Tuple[int, float, LineSpectrum]]:
    # Lazily import baseline line localization to avoid circular imports
    from line_localization_baseline import test_cases_from_hdd_result, collect_line_coverage

    test_cases = test_cases_from_hdd_result(result, debugger)
    coverage = collect_line_coverage(program_path, test_cases, entry_function=entry_function)
    return rank_lines(coverage, formula=formula, normalize=normalize)


def render_report(program_path: str, ranking: List[Tuple[int, float, LineSpectrum]], top_n: Optional[int] = None) -> str:
    with open(program_path, "r", encoding="utf-8") as f:
        source_lines = f.readlines()

    ranked = ranking if top_n is None else ranking[:top_n]
    rows = []
    for line_no, score, spec in ranked:
        text = source_lines[line_no - 1].rstrip("\n") if 0 < line_no <= len(source_lines) else ""
        rows.append(f"  {score:6.3f}  L{line_no:<4} ef={spec.ef:g} ep={spec.ep:g} nf={spec.nf:g} np={spec.np:g}   {text}")
    header = f"Suspiciousness report for {program_path}"
    return header + "\n" + "\n".join(rows)


def compare_rankings(baseline_ranking: List[Tuple[int, float, LineSpectrum]], weighted_ranking: List[Tuple[int, float, Any]], faulty_lines: List[int], top_ns: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
    # Produce simple top-N accuracy and EXAM-like metrics comparing two rankings.
    def top_n_set(ranking, n):
        return {line for line, _, _ in ranking[:n]}

    results: Dict[str, Any] = {}
    for n in top_ns:
        base_top = top_n_set(baseline_ranking, n)
        weight_top = top_n_set(weighted_ranking, n)
        results[f"baseline_top_{n}"] = sum(1 for f in faulty_lines if f in base_top) / len(faulty_lines) if faulty_lines else 0.0
        results[f"weighted_top_{n}"] = sum(1 for f in faulty_lines if f in weight_top) / len(faulty_lines) if faulty_lines else 0.0

    # First faulty line rank (lower is better)
    def first_rank(ranking, faulty):
        for idx, (line, _, _) in enumerate(ranking, start=1):
            if line in faulty:
                return idx
        return None

    results["baseline_first_ranks"] = [first_rank(baseline_ranking, [f]) for f in faulty_lines]
    results["weighted_first_ranks"] = [first_rank(weighted_ranking, [f]) for f in faulty_lines]

    return results


if __name__ == "__main__":
    # Quick smoke test: import baseline modules and run demo flow.
    from hdd_baseline import HierarchicalDeltaDebugger
    from line_localization_baseline import test_cases_from_hdd_result, collect_line_coverage

    sample_input = {"items": [{"value": 1}, {"value": 2}, {"value": "BUG"}, {"value": 3}]}
    program_path = "sample_buggy_program.py"

    def oracle(candidate: Any) -> bool:
        return any(v == "BUG" for v in (candidate.get("items") if isinstance(candidate, dict) else [])) if isinstance(candidate, dict) else False

    debugger = HierarchicalDeltaDebugger(oracle=oracle)
    result = debugger.reduce(sample_input)
    test_cases = test_cases_from_hdd_result(result, debugger)
    coverage = collect_line_coverage(program_path, test_cases)
    ranking = rank_lines(coverage)
    print(render_report(program_path, ranking, top_n=5))
