"""Observed and deduplicated spectrum evidence for HDD-LOC."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, Optional, Sequence, Tuple

from execution_records import ExecutionRecord, Outcome, SourceLocation
from sbfl_score import calculate_sbfl_score


@dataclass(frozen=True)
class SpectrumEvidence:
    """One independent evidence item used by a spectrum calculation."""

    outcome: Outcome
    coverage: FrozenSet[SourceLocation]
    execution_ids: Tuple[int, ...]

    @property
    def multiplicity(self) -> int:
        return len(self.execution_ids)


@dataclass(frozen=True)
class ObservedSpectrum:
    """Unweighted, directly observed counts for one buggy source line."""

    location: SourceLocation
    ef: int
    ep: int
    nf: int
    np: int


@dataclass(frozen=True)
class ObservedLineScore:
    location: SourceLocation
    score: float
    spectrum: ObservedSpectrum


def select_execution_budget(
    records: Sequence[ExecutionRecord],
    budget: Optional[int],
) -> Tuple[ExecutionRecord, ...]:
    """Select a chronological prefix before validity filtering/deduplication."""

    if budget is None:
        return tuple(records)
    if budget < 0:
        raise ValueError("execution budget must not be negative")
    return tuple(records[:budget])


def build_evidence_pool(
    records: Sequence[ExecutionRecord],
    *,
    deduplicate: bool,
    budget: Optional[int] = None,
) -> Tuple[SpectrumEvidence, ...]:
    """Project immutable execution records into a passive evidence pool.

    INVALID records and records that failed validity checks remain in the
    original input sequence but do not enter this pool.
    """

    selected = select_execution_budget(records, budget)
    valid = [record for record in selected if record.contributes_spectrum_evidence]
    if not deduplicate:
        return tuple(
            SpectrumEvidence(
                outcome=record.outcome,
                coverage=record.coverage,
                execution_ids=(record.execution_id,),
            )
            for record in valid
        )

    grouped: dict[tuple[Outcome, FrozenSet[SourceLocation]], list[int]] = {}
    order: list[tuple[Outcome, FrozenSet[SourceLocation]]] = []
    for record in valid:
        key = (record.outcome, record.coverage)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(record.execution_id)
    return tuple(
        SpectrumEvidence(
            outcome=outcome,
            coverage=coverage,
            execution_ids=tuple(grouped[(outcome, coverage)]),
        )
        for outcome, coverage in order
    )


def aggregate_observed_spectrum(
    evidence: Sequence[SpectrumEvidence],
    line_universe: Optional[Iterable[SourceLocation]] = None,
) -> Dict[SourceLocation, ObservedSpectrum]:
    """Calculate raw ef/ep/nf/np without priors or structural weights."""

    locations = set(line_universe or ())
    locations.update(location for item in evidence for location in item.coverage)
    total_failed = sum(1 for item in evidence if item.outcome is Outcome.FAIL)
    total_passed = sum(1 for item in evidence if item.outcome is Outcome.PASS)

    spectra: Dict[SourceLocation, ObservedSpectrum] = {}
    for location in sorted(locations):
        ef = sum(1 for item in evidence if item.outcome is Outcome.FAIL and location in item.coverage)
        ep = sum(1 for item in evidence if item.outcome is Outcome.PASS and location in item.coverage)
        spectra[location] = ObservedSpectrum(
            location=location,
            ef=ef,
            ep=ep,
            nf=total_failed - ef,
            np=total_passed - ep,
        )
    return spectra


def rank_observed_spectrum(
    spectra: Dict[SourceLocation, ObservedSpectrum],
    formula: str = "ochiai",
    normalize: bool = True,
) -> list[ObservedLineScore]:
    scored = [
        ObservedLineScore(
            location=location,
            score=calculate_sbfl_score(spectrum.ef, spectrum.ep, spectrum.nf, spectrum.np, formula),
            spectrum=spectrum,
        )
        for location, spectrum in spectra.items()
    ]
    if normalize and scored:
        finite = [item.score for item in scored if math.isfinite(item.score)]
        maximum = max(finite) if finite else 1.0
        maximum = maximum or 1.0
        scored = [
            ObservedLineScore(
                location=item.location,
                score=item.score if not math.isfinite(item.score) else item.score / maximum,
                spectrum=item.spectrum,
            )
            for item in scored
        ]
    scored.sort(key=lambda item: (-item.score, item.location.filename, item.location.line))
    return scored
