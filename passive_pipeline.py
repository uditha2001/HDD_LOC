"""Ablation views over one immutable HDD execution suite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from bayesian_sbfl import (
    BetaPrior,
    MonteCarloLineScore,
    PosteriorMeanLineScore,
    estimate_all_posteriors,
    monte_carlo_ranking,
    rank_posterior_means,
)
from execution_records import ExecutionRecord
from spectrum_evidence import (
    ObservedLineScore,
    aggregate_observed_spectrum,
    build_evidence_pool,
    rank_observed_spectrum,
)


DEFAULT_EXECUTION_BUDGETS: Tuple[Optional[int], ...] = (5, 10, 20, 30, 50, None)


@dataclass(frozen=True)
class PassiveAnalysis:
    """A-D ablation results for one formula and one execution budget."""

    budget: Optional[int]
    formula: str
    original_execution_count: int
    raw_evidence_count: int
    deduplicated_evidence_count: int
    raw_ranking: Tuple[ObservedLineScore, ...]
    deduplicated_raw_ranking: Tuple[ObservedLineScore, ...]
    posterior_mean_ranking: Tuple[PosteriorMeanLineScore, ...]
    monte_carlo_ranking: Tuple[MonteCarloLineScore, ...]


def analyze_execution_records(
    records: Sequence[ExecutionRecord],
    *,
    formulas: Sequence[str] = ("ochiai", "jaccard"),
    budgets: Sequence[Optional[int]] = DEFAULT_EXECUTION_BUDGETS,
    prior: BetaPrior = BetaPrior(),
    monte_carlo_samples: int = 5_000,
    random_seed: int = 0,
) -> Tuple[PassiveAnalysis, ...]:
    """Run A-D without regenerating or mutating the underlying executions."""

    analyses = []
    for budget in budgets:
        raw_evidence = build_evidence_pool(records, deduplicate=False, budget=budget)
        deduplicated = build_evidence_pool(records, deduplicate=True, budget=budget)
        raw_spectra = aggregate_observed_spectrum(raw_evidence)
        deduplicated_spectra = aggregate_observed_spectrum(deduplicated)
        posteriors = estimate_all_posteriors(deduplicated_spectra, prior)
        original_count = len(records) if budget is None else min(budget, len(records))

        for formula in formulas:
            analyses.append(
                PassiveAnalysis(
                    budget=budget,
                    formula=formula,
                    original_execution_count=original_count,
                    raw_evidence_count=len(raw_evidence),
                    deduplicated_evidence_count=len(deduplicated),
                    raw_ranking=tuple(rank_observed_spectrum(raw_spectra, formula)),
                    deduplicated_raw_ranking=tuple(
                        rank_observed_spectrum(deduplicated_spectra, formula)
                    ),
                    posterior_mean_ranking=tuple(
                        rank_posterior_means(posteriors, formula)
                    ),
                    monte_carlo_ranking=tuple(
                        monte_carlo_ranking(
                            deduplicated_spectra,
                            formula,
                            prior=prior,
                            samples=monte_carlo_samples,
                            seed=random_seed,
                        )
                    ),
                )
            )
    return tuple(analyses)
