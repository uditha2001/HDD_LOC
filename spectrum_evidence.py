"""Observed and deduplicated spectrum evidence for HDD-LOC."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, Literal, Optional, Sequence, Tuple

from execution_records import ExecutionRecord, Outcome, SourceLocation
from sbfl_score import calculate_sbfl_score


EvidenceWeighting = Literal["unit", "inverse_hdd"]


@dataclass(frozen=True)
class SpectrumEvidence:
    """One independent evidence item used by a spectrum calculation."""

    outcome: Outcome
    coverage: FrozenSet[SourceLocation]
    execution_ids: Tuple[int, ...]
    member_weights: Tuple[float, ...]
    evidence_weight: float

    @property
    def multiplicity(self) -> int:
        return len(self.execution_ids)


@dataclass(frozen=True)
class ObservedSpectrum:
    """Directly observed evidence mass for one buggy source line.

    Values are integer counts under ``unit`` weighting and fractional
    weighted counts under ``inverse_hdd`` weighting.
    """

    location: SourceLocation
    ef: float
    ep: float
    nf: float
    np: float


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
    weighting: EvidenceWeighting = "unit",
) -> Tuple[SpectrumEvidence, ...]:
    """Project immutable execution records into a passive evidence pool.

    INVALID records and records that failed validity checks remain in the
    original input sequence but do not enter this pool.
    """

    if weighting not in ("unit", "inverse_hdd"):
        raise ValueError(f"unknown evidence weighting: {weighting!r}")

    selected = select_execution_budget(records, budget)
    valid = [record for record in selected if record.contributes_spectrum_evidence]

    def record_weight(record: ExecutionRecord) -> float:
        weight = 1.0 if weighting == "unit" else record.inverse_hdd_weight
        if weight <= 0:
            raise ValueError(
                f"execution {record.execution_id} has non-positive evidence weight {weight}"
            )
        return weight

    if not deduplicate:
        return tuple(
            SpectrumEvidence(
                outcome=record.outcome,
                coverage=record.coverage,
                execution_ids=(record.execution_id,),
                member_weights=(record_weight(record),),
                evidence_weight=record_weight(record),
            )
            for record in valid
        )

    grouped: dict[
        tuple[Outcome, FrozenSet[SourceLocation]],
        list[tuple[int, float]],
    ] = {}
    order: list[tuple[Outcome, FrozenSet[SourceLocation]]] = []
    for record in valid:
        key = (record.outcome, record.coverage)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append((record.execution_id, record_weight(record)))
    return tuple(
        SpectrumEvidence(
            outcome=outcome,
            coverage=coverage,
            execution_ids=tuple(
                execution_id for execution_id, _ in grouped[(outcome, coverage)]
            ),
            member_weights=tuple(
                weight for _, weight in grouped[(outcome, coverage)]
            ),
            # A duplicate group is one canonical observation. Averaging retains
            # its typical HDD confidence without summing dependent evidence.
            evidence_weight=sum(
                weight for _, weight in grouped[(outcome, coverage)]
            )
            / len(grouped[(outcome, coverage)]),
        )
        for outcome, coverage in order
    )


def aggregate_observed_spectrum(
    evidence: Sequence[SpectrumEvidence],
    line_universe: Optional[Iterable[SourceLocation]] = None,
) -> Dict[SourceLocation, ObservedSpectrum]:
    """Calculate directly observed ef/ep/nf/np evidence mass."""

    locations = set(line_universe or ())
    locations.update(location for item in evidence for location in item.coverage)
    total_failed = sum(
        item.evidence_weight for item in evidence if item.outcome is Outcome.FAIL
    )
    total_passed = sum(
        item.evidence_weight for item in evidence if item.outcome is Outcome.PASS
    )

    spectra: Dict[SourceLocation, ObservedSpectrum] = {}
    for location in sorted(locations):
        ef = sum(
            item.evidence_weight
            for item in evidence
            if item.outcome is Outcome.FAIL and location in item.coverage
        )
        ep = sum(
            item.evidence_weight
            for item in evidence
            if item.outcome is Outcome.PASS and location in item.coverage
        )
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
