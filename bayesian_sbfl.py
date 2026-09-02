"""Bayesian spectrum estimation and uncertainty propagation for HDD-LOC."""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from dataclasses import dataclass
from typing import Dict, Sequence

from execution_records import SourceLocation
from sbfl_score import calculate_sbfl_score
from spectrum_evidence import ObservedSpectrum


@dataclass(frozen=True)
class BetaPrior:
    alpha: float = 0.5
    beta: float = 0.5

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("Beta prior parameters must be positive")


@dataclass(frozen=True)
class PosteriorSpectrum:
    """Observed counts and deterministic posterior-mean estimates."""

    location: SourceLocation
    observed: ObservedSpectrum
    mean_p1: float
    mean_p0: float
    estimated_ef: float
    estimated_ep: float
    estimated_nf: float
    estimated_np: float
    necessity_alpha: float
    necessity_beta: float
    necessity_mean: float


@dataclass(frozen=True)
class PosteriorMeanLineScore:
    location: SourceLocation
    score: float
    posterior: PosteriorSpectrum


@dataclass(frozen=True)
class MonteCarloLineScore:
    location: SourceLocation
    formula: str
    posterior_mean: float
    posterior_median: float
    standard_deviation: float
    credible_interval_low: float
    credible_interval_high: float
    sample_count: int
    prior: BetaPrior
    observed: ObservedSpectrum


def estimate_posterior_spectrum(
    observed: ObservedSpectrum,
    prior: BetaPrior = BetaPrior(),
) -> PosteriorSpectrum:
    covered_total = observed.ef + observed.ep
    uncovered_total = observed.nf + observed.np

    mean_p1 = (prior.alpha + observed.ef) / (
        prior.alpha + prior.beta + covered_total
    )
    mean_p0 = (prior.alpha + observed.nf) / (
        prior.alpha + prior.beta + uncovered_total
    )
    necessity_alpha = prior.alpha + observed.ef
    necessity_beta = prior.beta + observed.nf
    necessity_mean = necessity_alpha / (necessity_alpha + necessity_beta)

    return PosteriorSpectrum(
        location=observed.location,
        observed=observed,
        mean_p1=mean_p1,
        mean_p0=mean_p0,
        estimated_ef=covered_total * mean_p1,
        estimated_ep=covered_total * (1.0 - mean_p1),
        estimated_nf=uncovered_total * mean_p0,
        estimated_np=uncovered_total * (1.0 - mean_p0),
        necessity_alpha=necessity_alpha,
        necessity_beta=necessity_beta,
        necessity_mean=necessity_mean,
    )


def estimate_all_posteriors(
    spectra: Dict[SourceLocation, ObservedSpectrum],
    prior: BetaPrior = BetaPrior(),
) -> Dict[SourceLocation, PosteriorSpectrum]:
    return {
        location: estimate_posterior_spectrum(spectrum, prior)
        for location, spectrum in spectra.items()
    }


def rank_posterior_means(
    posteriors: Dict[SourceLocation, PosteriorSpectrum],
    formula: str = "ochiai",
) -> list[PosteriorMeanLineScore]:
    ranked = [
        PosteriorMeanLineScore(
            location=location,
            score=calculate_sbfl_score(
                posterior.estimated_ef,
                posterior.estimated_ep,
                posterior.estimated_nf,
                posterior.estimated_np,
                formula,
            ),
            posterior=posterior,
        )
        for location, posterior in posteriors.items()
    ]
    ranked.sort(key=lambda item: (-item.score, item.location.filename, item.location.line))
    return ranked


def _location_seed(seed: int, location: SourceLocation) -> int:
    material = f"{seed}\0{location.filename}\0{location.line}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot calculate a quantile of no values")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def monte_carlo_line_score(
    observed: ObservedSpectrum,
    formula: str = "ochiai",
    *,
    prior: BetaPrior = BetaPrior(),
    samples: int = 5_000,
    seed: int = 0,
) -> MonteCarloLineScore:
    if samples <= 0:
        raise ValueError("Monte Carlo sample count must be positive")

    covered_total = observed.ef + observed.ep
    uncovered_total = observed.nf + observed.np
    rng = random.Random(_location_seed(seed, observed.location))
    scores: list[float] = []
    for _ in range(samples):
        p1 = rng.betavariate(prior.alpha + observed.ef, prior.beta + observed.ep)
        p0 = rng.betavariate(prior.alpha + observed.nf, prior.beta + observed.np)
        ef = covered_total * p1
        ep = covered_total * (1.0 - p1)
        nf = uncovered_total * p0
        np_ = uncovered_total * (1.0 - p0)
        score = calculate_sbfl_score(ef, ep, nf, np_, formula)
        if not math.isfinite(score):
            raise ValueError(
                f"formula {formula!r} produced a non-finite Monte Carlo score "
                f"for {observed.location.filename}:{observed.location.line}"
            )
        scores.append(score)

    ordered = sorted(scores)
    return MonteCarloLineScore(
        location=observed.location,
        formula=formula,
        posterior_mean=statistics.fmean(scores),
        posterior_median=statistics.median(ordered),
        standard_deviation=statistics.pstdev(scores),
        credible_interval_low=_quantile(ordered, 0.025),
        credible_interval_high=_quantile(ordered, 0.975),
        sample_count=samples,
        prior=prior,
        observed=observed,
    )


def monte_carlo_ranking(
    spectra: Dict[SourceLocation, ObservedSpectrum],
    formula: str = "ochiai",
    *,
    prior: BetaPrior = BetaPrior(),
    samples: int = 5_000,
    seed: int = 0,
) -> list[MonteCarloLineScore]:
    ranked = [
        monte_carlo_line_score(
            spectrum,
            formula,
            prior=prior,
            samples=samples,
            seed=seed,
        )
        for spectrum in spectra.values()
    ]
    ranked.sort(
        key=lambda item: (
            -item.posterior_mean,
            item.location.filename,
            item.location.line,
        )
    )
    return ranked
