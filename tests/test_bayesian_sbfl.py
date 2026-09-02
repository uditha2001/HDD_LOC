from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bayesian_sbfl import (
    BetaPrior,
    estimate_posterior_spectrum,
    monte_carlo_line_score,
)
from execution_records import SourceLocation
from sbfl_score import calculate_sbfl_score
from spectrum_evidence import ObservedSpectrum


def _spectrum(ef, ep, nf, np):
    location = SourceLocation("engine.py", 211)
    return ObservedSpectrum(location=location, ef=ef, ep=ep, nf=nf, np=np)


def test_jeffreys_posterior_mean_calculation():
    posterior = estimate_posterior_spectrum(_spectrum(3, 1, 1, 5))

    assert posterior.mean_p1 == pytest.approx(3.5 / 5.0)
    assert posterior.mean_p0 == pytest.approx(1.5 / 7.0)
    assert posterior.estimated_ef == pytest.approx(4.0 * 3.5 / 5.0)
    assert posterior.estimated_ep == pytest.approx(4.0 * 1.5 / 5.0)
    assert posterior.estimated_nf == pytest.approx(6.0 * 1.5 / 7.0)
    assert posterior.estimated_np == pytest.approx(6.0 * 5.5 / 7.0)
    assert posterior.necessity_mean == pytest.approx(3.5 / 5.0)


def test_all_fail_observations_remain_observed_as_all_fail():
    observed = _spectrum(3, 0, 2, 0)
    posterior = estimate_posterior_spectrum(observed)

    assert posterior.observed.ep == 0
    assert posterior.observed.np == 0
    assert posterior.necessity_mean == pytest.approx(3.5 / 6.0)
    # These are explicitly inferred posterior cells, not observed PASS tests.
    assert posterior.estimated_ep > 0
    assert posterior.estimated_np > 0


def test_prior_is_configurable_and_must_be_positive():
    posterior = estimate_posterior_spectrum(_spectrum(1, 1, 1, 1), BetaPrior(2.0, 3.0))

    assert posterior.mean_p1 == pytest.approx(3.0 / 7.0)
    with pytest.raises(ValueError):
        BetaPrior(alpha=0.0, beta=0.5)


@pytest.mark.parametrize("formula", ["ochiai", "jaccard"])
def test_monte_carlo_is_seeded_and_reports_a_credible_interval(formula):
    observed = _spectrum(3, 1, 1, 5)

    first = monte_carlo_line_score(observed, formula, samples=800, seed=37)
    second = monte_carlo_line_score(observed, formula, samples=800, seed=37)
    different_seed = monte_carlo_line_score(observed, formula, samples=800, seed=38)

    assert first == second
    assert first.posterior_mean != different_seed.posterior_mean
    assert first.credible_interval_low <= first.posterior_median <= first.credible_interval_high
    assert first.standard_deviation > 0
    assert first.sample_count == 800


def test_public_formula_function_preserves_existing_ochiai_math():
    expected = 3.0 / math.sqrt((3.0 + 1.0) * (3.0 + 2.0))
    assert calculate_sbfl_score(3.0, 2.0, 1.0, 4.0, "ochiai") == pytest.approx(expected)
