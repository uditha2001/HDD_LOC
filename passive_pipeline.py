"""Raw SBFL views over one immutable HDD execution suite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from execution_records import ExecutionRecord
from spectrum_evidence import (
    EvidenceWeighting,
    ObservedLineScore,
    aggregate_observed_spectrum,
    build_evidence_pool,
    rank_observed_spectrum,
)


DEFAULT_EXECUTION_BUDGETS: Tuple[Optional[int], ...] = (5, 10, 20, 30, 50, None)


@dataclass(frozen=True)
class PassiveAnalysis:
    """Observed raw-SBFL results for one formula and execution budget."""

    budget: Optional[int]
    formula: str
    evidence_weighting: EvidenceWeighting
    original_execution_count: int
    raw_evidence_count: int
    deduplicated_evidence_count: int
    raw_ranking: Tuple[ObservedLineScore, ...]
    deduplicated_raw_ranking: Tuple[ObservedLineScore, ...]


def analyze_execution_records(
    records: Sequence[ExecutionRecord],
    *,
    formulas: Sequence[str] = ("ochiai", "jaccard", "tarantula", "dstar2"),
    budgets: Sequence[Optional[int]] = DEFAULT_EXECUTION_BUDGETS,
    evidence_weighting: EvidenceWeighting = "inverse_hdd",
) -> Tuple[PassiveAnalysis, ...]:
    """Rank observed spectra without mutating the underlying executions."""

    analyses = []
    for budget in budgets:
        raw_evidence = build_evidence_pool(
            records,
            deduplicate=False,
            budget=budget,
            weighting=evidence_weighting,
        )
        deduplicated = build_evidence_pool(
            records,
            deduplicate=True,
            budget=budget,
            weighting=evidence_weighting,
        )
        raw_spectra = aggregate_observed_spectrum(raw_evidence)
        deduplicated_spectra = aggregate_observed_spectrum(deduplicated)
        original_count = len(records) if budget is None else min(budget, len(records))

        for formula in formulas:
            analyses.append(
                PassiveAnalysis(
                    budget=budget,
                    formula=formula,
                    evidence_weighting=evidence_weighting,
                    original_execution_count=original_count,
                    raw_evidence_count=len(raw_evidence),
                    deduplicated_evidence_count=len(deduplicated),
                    raw_ranking=tuple(rank_observed_spectrum(raw_spectra, formula)),
                    deduplicated_raw_ranking=tuple(
                        rank_observed_spectrum(deduplicated_spectra, formula)
                    ),
                )
            )
    return tuple(analyses)
