"""Hierarchical Delta Debugging (HDD-LOC, minimization stage).

This module's one job is: given a failing structured input (nested
lists/tuples/dicts/sets) and an oracle, minimize the input, and return
every test case the minimization process ran along the way -- each test
case tagged with whether it passed or failed and the WDD weight of the
partition/element that test was evaluating.

That test-case list (candidate input, pass/fail, weight) is the complete
output of this module. Turning it into fault-localization suspiciousness
scores (SBFL) is a separate concern, handled by another module.

Reduction follows Weighted Delta Debugging (WDD, Zhou et al., ICSE 2025):
instead of removing one element at a time, each container's children are
split into two partitions whose *weights* (not counts) are as close to
equal as possible, tests are run against a partition alone and against its
complement, and the search re-partitions only when neither succeeds. A
finishing ``ensureOneMinimal`` pass (single-element removal to a fixpoint)
guarantees 1-tree-minimality afterward. Weight defaults to subtree size
(leaf count) as the structural analogue of WDD's token count, but is
swappable (``weighting="uniform"`` reduces to classic, unweighted
ddmin-style partitioning -- useful as an ablation baseline; ``weighting=
"custom"`` accepts a caller-supplied ``weight_fn``).

Design notes
------------
- Every node in the input tree (containers *and* leaves) gets a stable
  integer ``id`` assigned once, at tree-construction time. Reduction only
  ever *removes* ids from an ``active_ids`` set -- it never renumbers
  anything -- so a node's id remains a valid key for the whole run, unlike
  a path tuple, which shifts as siblings are removed.
- A candidate under test is materialized on demand from ``active_ids``, so
  the coverage information (which ids were active), the oracle's
  pass/fail verdict, and the test's WDD weight are all recorded together
  in a single ``TestRecord``.
- The oracle is injected: it must return True for inputs that still
  reproduce the failure.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple, Union

Kind = str  # one of: 'list', 'tuple', 'dict', 'set', 'leaf'
PathKey = Union[str, int]


# --------------------------------------------------------------------------
# Tree representation
# --------------------------------------------------------------------------


@dataclass
class Node:
    """One node of the structured-input tree, with a stable identity.

    For containers, ``children`` (list/tuple/set) or ``dict_items`` (dict)
    hold the child nodes. For leaves, ``value`` holds the original value and
    ``children``/``dict_items`` are empty.
    """

    id: int
    path: Tuple[PathKey, ...]
    depth: int
    kind: Kind
    value: Any = None
    children: List["Node"] = field(default_factory=list)
    dict_items: List[Tuple[Any, "Node"]] = field(default_factory=list)

    def child_nodes(self) -> List["Node"]:
        if self.kind == "dict":
            return [child for _, child in self.dict_items]
        return self.children


# --------------------------------------------------------------------------
# Test records and results
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TestRecord:
    """One property test HDD ran during minimization."""

    test_id: int
    active_ids: FrozenSet[int]  # every node id present in this test's candidate
    failed: bool  # oracle verdict: True = still reproduces the failure
    weight: float  # WDD weight of the partition/element this test evaluated
    trial_node_id: Optional[int]  # single node this test isolates; None for multi-element tests / baseline
    trial_depth: Optional[int]  # depth of the container level this test was run at


@dataclass(frozen=True)
class HDDResult:
    """Minimized failing input plus every test case run to get there."""

    minimal_failing_input: Any
    test_records: Tuple[TestRecord, ...]
    nodes: Dict[int, Node]


# --------------------------------------------------------------------------
# Core debugger
# --------------------------------------------------------------------------


class HierarchicalDeltaDebugger:
    """Minimize a structured failing input with hierarchical, WDD-weighted
    delta debugging, recording every test case (candidate, pass/fail,
    weight) run along the way.
    """

    def __init__(
        self,
        oracle: Optional[Callable[[Any], bool]] = None,
        weighting: str = "subtree_size",
        weight_fn: Optional[Callable[[Node], float]] = None,
    ) -> None:
        """
        weighting: how each element's WDD weight is computed during
            partitioning.
            - 'subtree_size' (default): weight = number of leaves in the
              node's subtree -- the structural analogue of WDD's token
              count. Reproduces Wddmin's weighted-partitioning behavior.
            - 'uniform': every element weighs 1, so partitioning splits
              evenly by count instead of by size -- reproduces classic,
              unweighted ddmin partitioning. Useful as an ablation baseline
              against 'subtree_size'.
            - 'custom': use the caller-supplied weight_fn(node) -> number,
              e.g. real token counts if the original source spans are
              available.
        weight_fn: required when weighting='custom'; ignored otherwise.
        """

        self.oracle = oracle
        self.weighting = weighting
        self._custom_weight_fn = weight_fn

        # Populated fresh by each call to reduce().
        self.root: Optional[Node] = None
        self.nodes: Dict[int, Node] = {}
        self._subtree_ids: Dict[int, FrozenSet[int]] = {}
        self._element_weight: Dict[int, float] = {}
        self.active_ids: Set[int] = set()
        self._records: List[TestRecord] = []
        self._id_counter = itertools.count()
        self._test_counter = itertools.count()

    # -- public API --------------------------------------------------

    def reduce(self, structured_input: Any) -> HDDResult:
        """Minimize structured_input and return it plus every test case run.

        The oracle must return True for inputs that still fail.
        """

        if self.oracle is None:
            raise NotImplementedError("Provide a test oracle that returns True for failing inputs.")

        self._id_counter = itertools.count()
        self._test_counter = itertools.count()
        self._records = []
        self.nodes = {}
        self._element_weight = {}

        self.root = self._build_tree(structured_input, path=(), depth=0)
        self._index_nodes(self.root)
        self._subtree_ids = {nid: self._collect_subtree_ids(node) for nid, node in self.nodes.items()}
        self.active_ids = set(self.nodes.keys())

        baseline_weight = self._weight_of(self.root)
        baseline_failed = self._run_test(
            self.active_ids, trial_node_id=None, trial_depth=None, weight=baseline_weight
        )
        if not baseline_failed:
            # The un-reduced input doesn't reproduce the failure. Still return
            # it as-is rather than reducing against a passing baseline, which
            # would silently produce meaningless test cases.
            return HDDResult(
                minimal_failing_input=structured_input,
                test_records=tuple(self._records),
                nodes=dict(self.nodes),
            )

        self._reduce_node(self.root)

        minimal = self._materialize(self.root, self.active_ids)
        return HDDResult(
            minimal_failing_input=minimal,
            test_records=tuple(self._records),
            nodes=dict(self.nodes),
        )

    # -- tree construction ---------------------------------------------

    def _next_id(self) -> int:
        return next(self._id_counter)

    def _build_tree(self, value: Any, path: Tuple[PathKey, ...], depth: int) -> Node:
        node_id = self._next_id()
        if isinstance(value, list):
            children = [self._build_tree(v, path + (i,), depth + 1) for i, v in enumerate(value)]
            return Node(id=node_id, path=path, depth=depth, kind="list", children=children)
        if isinstance(value, tuple):
            children = [self._build_tree(v, path + (i,), depth + 1) for i, v in enumerate(value)]
            return Node(id=node_id, path=path, depth=depth, kind="tuple", children=children)
        if isinstance(value, dict):
            items = [(k, self._build_tree(v, path + (k,), depth + 1)) for k, v in value.items()]
            return Node(id=node_id, path=path, depth=depth, kind="dict", dict_items=items)
        if isinstance(value, set):
            ordered = sorted(value, key=repr)
            children = [self._build_tree(v, path + (i,), depth + 1) for i, v in enumerate(ordered)]
            return Node(id=node_id, path=path, depth=depth, kind="set", children=children)
        return Node(id=node_id, path=path, depth=depth, kind="leaf", value=value)

    def _index_nodes(self, node: Node) -> None:
        self.nodes[node.id] = node
        for child in node.child_nodes():
            self._index_nodes(child)

    def _collect_subtree_ids(self, node: Node) -> FrozenSet[int]:
        ids = {node.id}
        for child in node.child_nodes():
            ids |= self._subtree_ids.get(child.id) or self._collect_subtree_ids(child)
        return frozenset(ids)

    # -- materialization --------------------------------------------------

    def _materialize(self, node: Node, active_ids: Set[int]) -> Any:
        if node.kind == "leaf":
            return node.value
        if node.kind == "list":
            return [self._materialize(c, active_ids) for c in node.children if c.id in active_ids]
        if node.kind == "tuple":
            return tuple(self._materialize(c, active_ids) for c in node.children if c.id in active_ids)
        if node.kind == "set":
            return {self._materialize(c, active_ids) for c in node.children if c.id in active_ids}
        if node.kind == "dict":
            return {k: self._materialize(c, active_ids) for k, c in node.dict_items if c.id in active_ids}
        raise AssertionError(f"unreachable node kind: {node.kind}")

    # -- test execution ---------------------------------------------------------

    def _run_test(
        self,
        active_ids: Set[int],
        trial_node_id: Optional[int],
        trial_depth: Optional[int],
        weight: float,
    ) -> bool:
        candidate = self._materialize(self.root, active_ids)
        failed = bool(self.oracle(candidate))
        self._records.append(
            TestRecord(
                test_id=next(self._test_counter),
                active_ids=frozenset(active_ids),
                failed=failed,
                weight=weight,
                trial_node_id=trial_node_id,
                trial_depth=trial_depth,
            )
        )
        return failed

    def _reduce_node(self, node: Node) -> None:
        # Reduce children first (bottom-up), then run Wddmin over this
        # container's children -- mirrors HDD applying ddmin at every level
        # of the tree, but with WDD's weighted partitioning strategy.
        children = node.child_nodes()
        for child in children:
            self._reduce_node(child)
        if children:
            self._wddmin_children(children)

    # -- element weighting (WDD) -----------------------------------------

    def _weight_of(self, node: Node) -> float:
        """WDD's per-element weight, memoized. Computed bottom-up so a
        parent's 'subtree_size' weight only needs its children's weights,
        which were already computed during the bottom-up reduction pass.
        """

        cached = self._element_weight.get(node.id)
        if cached is not None:
            return cached

        if self.weighting == "uniform":
            w: float = 1.0
        elif self.weighting == "custom":
            if self._custom_weight_fn is None:
                raise ValueError("weighting='custom' requires a weight_fn")
            w = max(1.0, float(self._custom_weight_fn(node)))
        elif self.weighting == "subtree_size":
            if node.kind == "leaf":
                w = 1.0
            else:
                w = max(1.0, sum(self._weight_of(c) for c in node.child_nodes()))
        else:
            raise ValueError(f"unknown weighting scheme: {self.weighting!r}")

        self._element_weight[node.id] = w
        return w

    # -- Wddmin: weighted partitioning (Algorithm 1 of the WDD paper) ----

    def _wddmin_children(self, children: List[Node]) -> None:
        active_children = [c for c in children if c.id in self.active_ids]
        if len(active_children) <= 1:
            return  # nothing to partition

        children_by_id = {c.id: c for c in active_children}
        lmin_ids = self._wdd_reduce(children_by_id)
        lmin_ids = self._ensure_one_minimal(lmin_ids, children_by_id)

        keep = set(lmin_ids)
        for cid in children_by_id:
            if cid not in keep:
                self.active_ids -= self._subtree_ids[cid]

    def _wdd_reduce(self, children_by_id: Dict[int, Node]) -> List[int]:
        """Wddmin's main search: weighted binary partitioning until every
        surviving partition is a singleton, per the WDD paper's Algorithm 1.
        """

        all_ids = list(children_by_id.keys())
        lmin_ids = list(all_ids)
        partitions: List[List[int]] = [list(all_ids)]

        while partitions:
            progressed = False

            # Step 1: does any single partition alone still fail?
            for ptn in partitions:
                if self._run_container_test(children_by_id, all_ids, ptn):
                    lmin_ids = list(ptn)
                    partitions = self._weighted_partition([ptn], children_by_id)
                    progressed = True
                    break
            if progressed:
                continue

            # Step 2: does some partition's complement (within lmin) still fail?
            for ptn in partitions:
                complement = [i for i in lmin_ids if i not in ptn]
                if complement and self._run_container_test(children_by_id, all_ids, complement):
                    lmin_ids = complement
                    partitions = [p for p in partitions if p is not ptn]
                    progressed = True
                    break
            if progressed:
                continue

            # Step 3: neither worked -- split every partition further by weight.
            partitions = self._weighted_partition(partitions, children_by_id)

        return lmin_ids

    def _weighted_partition(self, partitions: List[List[int]], children_by_id: Dict[int, Node]) -> List[List[int]]:
        """Split each partition of size > 1 into two, choosing the split
        point whose weight sums are as close to equal as possible (WDD's
        weightedPartition; weighting='uniform' makes this equivalent to
        classic ddmin's even-by-count split).
        """

        result: List[List[int]] = []
        for ptn in partitions:
            if len(ptn) <= 1:
                continue  # can't split further -- drop, matching the paper
            weights = [self._weight_of(children_by_id[i]) for i in ptn]
            half = 0.5 * sum(weights)
            split_idx = self._closest_weighted_split(weights, half)
            p1, p2 = ptn[:split_idx], ptn[split_idx:]
            if p1:
                result.append(p1)
            if p2:
                result.append(p2)
        return result

    @staticmethod
    def _closest_weighted_split(weights: List[float], half: float) -> int:
        cumulative = 0.0
        best_idx = 1
        best_diff = float("inf")
        for i in range(1, len(weights)):
            cumulative += weights[i - 1]
            diff = abs(cumulative - half)
            if diff < best_diff:
                best_diff = diff
                best_idx = i
        return best_idx

    def _ensure_one_minimal(self, lmin_ids: List[int], children_by_id: Dict[int, Node]) -> List[int]:
        """Finishing pass: try removing each surviving element individually,
        to a fixpoint, guaranteeing 1-tree-minimality regardless of what the
        weighted partitioning search happened to try.
        """

        all_ids = list(children_by_id.keys())
        changed = True
        while changed:
            changed = False
            for cid in list(lmin_ids):
                candidate = [i for i in lmin_ids if i != cid]
                if not candidate:
                    continue  # never test an empty container
                if self._run_container_test(children_by_id, all_ids, candidate):
                    lmin_ids = candidate
                    changed = True
                    break
        return lmin_ids

    def _run_container_test(self, children_by_id: Dict[int, Node], all_ids: List[int], keep_ids: List[int]) -> bool:
        """Run one property test: within this container, keep exactly the
        children in keep_ids (and their subtrees); everything outside this
        container stays exactly as active_ids already has it. The test's
        weight is the total WDD weight of the elements being evaluated
        (keep_ids), so downstream SBFL can weight this test's evidence
        by how much of the input it actually targeted.
        """

        remove_all: Set[int] = set()
        for cid in all_ids:
            remove_all |= self._subtree_ids[cid]
        base_active = self.active_ids - remove_all

        keep_subtree_ids: Set[int] = set()
        for cid in keep_ids:
            keep_subtree_ids |= self._subtree_ids[cid]

        trial_active = base_active | keep_subtree_ids
        trial_node_id = keep_ids[0] if len(keep_ids) == 1 else None
        trial_depth = children_by_id[all_ids[0]].depth if all_ids else None
        weight = sum(self._weight_of(children_by_id[cid]) for cid in keep_ids)
        return self._run_test(trial_active, trial_node_id=trial_node_id, trial_depth=trial_depth, weight=weight)


# --------------------------------------------------------------------------
# Convenience helpers
# --------------------------------------------------------------------------


def run_hdd(structured_input: Any, oracle: Callable[[Any], bool]) -> HDDResult:
    """Convenience helper for one-off HDD runs."""

    debugger = HierarchicalDeltaDebugger(oracle=oracle)
    return debugger.reduce(structured_input)


def empty_oracle(_: Any) -> bool:
    """Placeholder oracle for callers that want to wire their own test logic later."""

    raise NotImplementedError("Replace empty_oracle with a real test oracle.")


if __name__ == "__main__":
    # Small demo: a nested structure where a leaf value "BUG" at depth 3
    # triggers the failure, buried inside several innocent siblings.
    sample_input = {
        "a": [1, 2, {"x": "ok", "y": "BUG"}],
        "b": [3, 4, 5],
        "c": {"nested": ["fine", "also fine"]},
    }

    def oracle(candidate: Any) -> bool:
        def contains_bug(value: Any) -> bool:
            if value == "BUG":
                return True
            if isinstance(value, dict):
                return any(contains_bug(v) for v in value.values())
            if isinstance(value, (list, tuple, set)):
                return any(contains_bug(v) for v in value)
            return False

        return contains_bug(candidate)

    for label, weighting in [("uniform (classic ddmin-style)", "uniform"), ("subtree_size (Wddmin-style)", "subtree_size")]:
        debugger = HierarchicalDeltaDebugger(oracle=oracle, weighting=weighting)
        result = debugger.reduce(sample_input)
        print(f"[{label}] minimal input: {result.minimal_failing_input}  "
              f"({len(result.test_records)} tests)")

    print()
    debugger = HierarchicalDeltaDebugger(oracle=oracle, weighting="subtree_size")
    result = debugger.reduce(sample_input)

    print("Test cases returned (candidate coverage, pass/fail, weight):")
    for record in result.test_records:
        print(
            f"  test_id={record.test_id:<3} failed={record.failed!s:<5} "
            f"weight={record.weight:<4g} trial_depth={record.trial_depth} "
            f"n_active_nodes={len(record.active_ids)}"
        )