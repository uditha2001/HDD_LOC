"""Line-level SBFL scoring for HDD-LOC.

This module consumes the line-coverage records produced by line_localization.py,
builds a line-based spectrum, and ranks the most suspicious lines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from hdd_loc import HDDResult
from line_localization import LineCoverageRecord, collect_line_coverage, test_cases_from_hdd_result


@dataclass(frozen=True)
class LineSpectrum:
    """Weighted spectrum counts for one source line."""

    line: int
    ef: float
    ep: float
    nf: float
    np: float


@dataclass(frozen=True)
class LineSuspiciousness:
    """A scored line with its supporting spectrum."""

    line: int
    spectrum: LineSpectrum
    score: float


def aggregate_line_spectrum(records: List[LineCoverageRecord]) -> Dict[int, LineSpectrum]:
    """Build the weighted spectrum for every line touched by any test."""

    all_lines = {line for record in records for line in record.lines}
    total_failed_weight = sum(record.weight for record in records if record.failed)
    total_passed_weight = sum(record.weight for record in records if not record.failed)

    spectrum: Dict[int, LineSpectrum] = {}
    for line in all_lines:
        ef = sum(record.weight for record in records if record.failed and line in record.lines)
        ep = sum(record.weight for record in records if not record.failed and line in record.lines)
        nf = total_failed_weight - ef
        np_ = total_passed_weight - ep
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


def rank_lines(
    records: List[LineCoverageRecord],
    formula: str = "ochiai",
    normalize: bool = True,
) -> List[Tuple[int, float, LineSpectrum]]:
    """Rank lines by SBFL using the recorded coverage and HDD weights."""

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


def rank_lines_from_hdd_result(
    result: HDDResult,
    debugger,
    program_path: str,
    entry_function: str = "run",
    formula: str = "ochiai",
    normalize: bool = True,
) -> List[Tuple[int, float, LineSpectrum]]:
    """Convenience wrapper that turns an HDD result into a line ranking."""

    test_cases = test_cases_from_hdd_result(result, debugger)
    coverage = collect_line_coverage(program_path, test_cases, entry_function=entry_function)
    return rank_lines(coverage, formula=formula, normalize=normalize)


def render_report(program_path: str, ranking: List[Tuple[int, float, LineSpectrum]], top_n: Optional[int] = None) -> str:
    """Render a human-readable, source-annotated suspiciousness report."""

    with open(program_path, "r", encoding="utf-8") as f:
        source_lines = f.readlines()

    ranked = ranking if top_n is None else ranking[:top_n]
    rows = []
    for line_no, score, spec in ranked:
        text = source_lines[line_no - 1].rstrip("\n") if 0 < line_no <= len(source_lines) else ""
        rows.append(
            f"  {score:6.3f}  L{line_no:<4} ef={spec.ef:g} ep={spec.ep:g} nf={spec.nf:g} np={spec.np:g}   {text}"
        )
    header = f"Suspiciousness report for {program_path}"
    return header + "\n" + "\n".join(rows)


if __name__ == "__main__":
    from hdd_loc import HierarchicalDeltaDebugger

    sample_input = {"items": [{"value": 1}, {"value": 2}, {"value": "BUG"}, {"value": 3}]}

    def oracle(candidate):
        return any(value == "BUG" for value in candidate.values()) if isinstance(candidate, dict) else False

    debugger = HierarchicalDeltaDebugger(oracle=oracle, weighting="subtree_size")
    result = debugger.reduce(sample_input)
    ranking = rank_lines_from_hdd_result(result, debugger, "sample_buggy_program.py")
    print(render_report("sample_buggy_program.py", ranking, top_n=5))
