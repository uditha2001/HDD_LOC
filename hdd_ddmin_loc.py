"""Baseline HDD whose per-container reducer is the original DDMIN-LOC core.

This variant deliberately reuses the baseline HDD tree, materialization, and
unit-weight execution-record semantics.  The only changed responsibility is
the sibling-set reducer: instead of baseline HDD's local count-partitioning
routine, it delegates membership minimization to ``ddmin_loc/DeltaMinimize``.

DDMIN-LOC is character based, so each active sibling is represented by one
private-use Unicode character.  The bridge maps a character subset returned
by the original algorithm back to the corresponding sibling node IDs before
the normal HDD oracle is executed.  No serialized program input is reduced at
this layer, and DDMIN-LOC's standalone SBFL/hybrid scripts are not invoked.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from ddmin_loc_adapter import _load_original_ddmin_module
from hdd_baseline import HierarchicalDeltaDebugger as BaselineHDD
from hdd_baseline import Node


_SYMBOL_START = 0xE000
_SYMBOL_CAPACITY = 0xF8FF - _SYMBOL_START + 1


class _SiblingSetOracle:
    """Adapt a DDMIN-LOC string candidate to one HDD sibling configuration."""

    def __init__(
        self,
        debugger: "HDDDDMinLocDebugger",
        children_by_id: Dict[int, Node],
        all_ids: List[int],
        symbol_to_id: Dict[str, int],
    ) -> None:
        self._debugger = debugger
        self._children_by_id = children_by_id
        self._all_ids = all_ids
        self._symbol_to_id = symbol_to_id

    def ask_human(self, test_input: Sequence[str] | str, _program: str) -> bool:
        # DeltaMinimize calls ``split('_')`` before this method. Our private
        # symbols never contain an underscore, so joining restores precisely
        # the candidate it constructed from sibling-membership characters.
        symbols = test_input if isinstance(test_input, str) else "".join(test_input)
        try:
            keep_ids = [self._symbol_to_id[symbol] for symbol in symbols]
        except KeyError as exc:  # defensive: a bridge bug must not look like PASS
            raise ValueError("DDMIN-LOC emitted an unknown HDD sibling symbol") from exc

        # Historical DDMIN-LOC convention: True means PASS/non-interesting;
        # False means the failure remains. HDD's internal oracle uses the
        # inverse convention, so invert exactly once at this boundary.
        return not self._debugger._run_container_test(
            self._children_by_id,
            self._all_ids,
            keep_ids,
        )


class HDDDDMinLocDebugger(BaselineHDD):
    """Baseline HDD with original DDMIN-LOC at every sibling-list reduction."""

    def _ddmin_children(self, children: List[Node]) -> None:
        active_children = [child for child in children if child.id in self.active_ids]
        if len(active_children) <= 1:
            return
        if len(active_children) > _SYMBOL_CAPACITY:
            raise ValueError(
                "HDD_DDmin_Loc supports at most "
                f"{_SYMBOL_CAPACITY} siblings in one container"
            )

        children_by_id = {child.id: child for child in active_children}
        all_ids = list(children_by_id)
        symbols = "".join(chr(_SYMBOL_START + offset) for offset in range(len(all_ids)))
        symbol_to_id = {
            symbol: child_id for symbol, child_id in zip(symbols, all_ids)
        }

        ddmin_module = _load_original_ddmin_module()
        bridge = _SiblingSetOracle(
            self,
            children_by_id,
            all_ids,
            symbol_to_id,
        )
        reducer = ddmin_module.DeltaMinimize(
            "hdd_ddmin_loc_siblings",
            bridge,
            False,
        )
        minimized_symbols, _passing, _failing = reducer.find_minimize_input(symbols)
        keep_ids = {symbol_to_id[symbol] for symbol in minimized_symbols}

        for child_id in all_ids:
            if child_id not in keep_ids:
                self.active_ids -= self._subtree_ids[child_id]


def run_hdd_ddmin_loc(structured_input, oracle):
    """Convenience helper for this reducer variant."""

    return HDDDDMinLocDebugger(oracle=oracle).reduce(structured_input)
