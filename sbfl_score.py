"""SBFL suspiciousness scoring for HDD-LOC.

Standalone module: it takes an ``hdd_loc.HDDResult`` (the output of
``HierarchicalDeltaDebugger.reduce()``) as input and computes spectrum-based
fault localization (SBFL) suspiciousness scores per input node.

This file is a one-way consumer of hdd_loc.py's output -- hdd_loc.py has no
knowledge of this file and never imports or calls it. hdd_loc.py's only job
is to minimize and hand back test cases (candidate coverage, pass/fail,
WDD weight); everything from here on -- turning those test cases into a
suspiciousness ranking -- lives in this file.

Two spectrum modes are supported:

- 'count'    : classic SBFL. ef/ep/nf/np are test *counts* per node --
               every test contributes equally to the spectrum, regardless
               of how much of the input it was exercising.
- 'weighted' : each test's contribution to ef/ep/nf/np is its WDD weight
               (hdd_loc.TestRecord.weight) instead of a flat 1. A test that
               isolated a large, heavily-weighted partition contributes
               more evidence than one that isolated a single leaf. This is
               the natural way to carry WDD's weighting concept into fault
               localization, rather than bolting on a separate ad hoc
               depth multiplier after the fact.

Four SBFL formulas are provided: Tarantula, Ochiai, Jaccard, and DStar2 --
DStar2 (Wong et al.) tends to outperform the other three in the SBFL
literature and is included as a stronger default option worth comparing
against Ochiai.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from hdd_loc import HDDResult, Node


# --------------------------------------------------------------------------
# Spectrum
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeSpectrum:
    """Spectrum counts (or weighted sums) for one input node."""

    node_id: int
    ef: float  # (weight of) failing tests where the node was present
    ep: float  # (weight of) passing tests where the node was present
    nf: float  # (weight of) failing tests where the node was absent
    np: float  # (weight of) passing tests where the node was absent


def compute_spectrum(result: HDDResult, mode: str = "weighted") -> Dict[int, NodeSpectrum]:
    """Build the per-node spectrum from every test case HDD-LOC ran.

    mode: 'count' for classic unweighted SBFL counts, or 'weighted' to use
    each test's WDD weight (hdd_loc.TestRecord.weight) instead of 1.
    """

    if mode not in ("count", "weighted"):
        raise ValueError(f"unknown spectrum mode: {mode!r}")

    def test_contribution(record) -> float:
        return record.weight if mode == "weighted" else 1.0

    total_failed = sum(test_contribution(r) for r in result.test_records if r.failed)
    total_passed = sum(test_contribution(r) for r in result.test_records if not r.failed)

    spectrum: Dict[int, NodeSpectrum] = {}
    for node_id in result.nodes:
        ef = ep = 0.0
        for record in result.test_records:
            if node_id not in record.active_ids:
                continue
            contribution = test_contribution(record)
            if record.failed:
                ef += contribution
            else:
                ep += contribution
        nf = total_failed - ef
        np_ = total_passed - ep
        spectrum[node_id] = NodeSpectrum(node_id=node_id, ef=ef, ep=ep, nf=nf, np=np_)
    return spectrum


# --------------------------------------------------------------------------
# SBFL formulas
# --------------------------------------------------------------------------


def sbfl_formula(ef: float, ep: float, nf: float, np_: float, formula: str) -> float:
    """Compute one SBFL suspiciousness value from spectrum counts.

    formula: 'tarantula' | 'ochiai' | 'jaccard' | 'dstar2'
    """

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
        return (ef ** 2) / denom

    raise ValueError(f"unknown SBFL formula: {formula!r}")


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Suspiciousness:
    """One node's SBFL suspiciousness score, with its supporting spectrum."""

    node: Node
    spectrum: NodeSpectrum
    score: float


def rank_nodes(
    result: HDDResult,
    formula: str = "ochiai",
    spectrum_mode: str = "weighted",
    normalize: bool = True,
) -> List[Suspiciousness]:
    """Score every node in result.nodes and return them ranked descending.

    formula: 'tarantula' | 'ochiai' | 'jaccard' | 'dstar2'
    spectrum_mode: 'count' | 'weighted' (see compute_spectrum)
    normalize: rescale scores into [0, 1] by dividing by the max (skipped
        for 'dstar2' when the max score is infinite -- ties at infinity are
        left as-is and simply sort to the top).
    """

    if not result.nodes:
        return []

    spectrum = compute_spectrum(result, mode=spectrum_mode)

    scored: List[Suspiciousness] = []
    for node_id, node in result.nodes.items():
        s = spectrum[node_id]
        score = sbfl_formula(s.ef, s.ep, s.nf, s.np, formula)
        scored.append(Suspiciousness(node=node, spectrum=s, score=score))

    if normalize and scored:
        finite_scores = [item.score for item in scored if math.isfinite(item.score)]
        max_score = max(finite_scores) if finite_scores else 1.0
        max_score = max_score or 1.0
        scored = [
            Suspiciousness(
                node=item.node,
                spectrum=item.spectrum,
                score=item.score if not math.isfinite(item.score) else item.score / max_score,
            )
            for item in scored
        ]

    scored.sort(key=lambda item: item.score, reverse=True)
    return scored


def render_report(ranking: List[Suspiciousness], top_n: Optional[int] = None) -> str:
    """Render a human-readable suspiciousness ranking."""

    rows = ranking if top_n is None else ranking[:top_n]
    lines = ["Suspiciousness report (HDD-LOC input nodes)"]
    for item in rows:
        s = item.spectrum
        lines.append(
            f"  {item.score:7.3f}  path={item.node.path!r:30} depth={item.node.depth} "
            f"kind={item.node.kind:6} ef={s.ef:g} ep={s.ep:g} nf={s.nf:g} np={s.np:g}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    from hdd_loc import HierarchicalDeltaDebugger

    sample_input = {
        "a": [1, 2, {"x": "ok", "y": "BUG"}],
        "b": [3, 4, 5],
        "c": {"nested": ["fine", "also fine"]},
    }

    def oracle(candidate):
        def contains_bug(value):
            if value == "BUG":
                return True
            if isinstance(value, dict):
                return any(contains_bug(v) for v in value.values())
            if isinstance(value, (list, tuple, set)):
                return any(contains_bug(v) for v in value)
            return False

        return contains_bug(candidate)

    debugger = HierarchicalDeltaDebugger(oracle=oracle, weighting="subtree_size")
    result = debugger.reduce(sample_input)

    print("Minimal failing input:", result.minimal_failing_input)
    print(f"Total test cases: {len(result.test_records)}\n")

    for mode in ("count", "weighted"):
        ranking = rank_nodes(result, formula="ochiai", spectrum_mode=mode)
        print(f"--- Ochiai, spectrum_mode={mode!r} ---")
        print(render_report(ranking, top_n=5))
        print()