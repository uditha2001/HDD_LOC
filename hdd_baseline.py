"""Traditional (unweighted) Hierarchical Delta Debugging baseline.

This file implements a baseline HDD that mirrors the structure and
APIs of the weighted `hdd_algorithm.py` but uses classic unweighted
ddmin-style partitioning (split by count) and records test cases with
unit weight (1.0). The implementation preserves node ids, materialization
behavior, and test-record layout so downstream replay is comparable.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple, Union

Kind = str
PathKey = Union[str, int]


@dataclass
class Node:
    id: int
    path: Tuple[PathKey, ...]
    depth: int
    kind: Kind
    value: Any = None
    children: List["Node"] = field(default_factory=list)
    dict_items: List[Tuple[Any, "Node"] ] = field(default_factory=list)

    def child_nodes(self) -> List["Node"]:
        if self.kind == "dict":
            return [child for _, child in self.dict_items]
        return self.children


@dataclass(frozen=True)
class TestRecord:
    """One property test HDD ran during minimization."""

    test_id: int
    active_ids: FrozenSet[int]
    failed: bool
    weight: float  # baseline uses unit weight (1.0)
    trial_node_id: Optional[int]
    trial_depth: Optional[int]


@dataclass(frozen=True)
class HDDResult:
    minimal_failing_input: Any
    test_records: Tuple[TestRecord, ...]
    nodes: Dict[int, Node]


class HierarchicalDeltaDebugger:
    """Unweighted HDD (baseline) using ddmin-style even partitions.

    The public API mirrors the weighted implementation: call `reduce()` to
    run minimization and inspect the returned `HDDResult` and `debugger` to
    reconstruct candidates.
    """

    def __init__(self, oracle: Optional[Callable[[Any], bool]] = None) -> None:
        self.oracle = oracle

        self.root: Optional[Node] = None
        self.nodes: Dict[int, Node] = {}
        self._subtree_ids: Dict[int, FrozenSet[int]] = {}
        self.active_ids: Set[int] = set()
        self._records: List[TestRecord] = []
        self._id_counter = itertools.count()
        self._test_counter = itertools.count()

    def reduce(self, structured_input: Any) -> HDDResult:
        if self.oracle is None:
            raise NotImplementedError("Provide a test oracle that returns True for failing inputs.")

        self._id_counter = itertools.count()
        self._test_counter = itertools.count()
        self._records = []
        self.nodes = {}

        self.root = self._build_tree(structured_input, path=(), depth=0)
        self._index_nodes(self.root)
        self._subtree_ids = {nid: self._collect_subtree_ids(node) for nid, node in self.nodes.items()}
        self.active_ids = set(self.nodes.keys())

        baseline_failed = self._run_test(self.active_ids, trial_node_id=None, trial_depth=None)
        if not baseline_failed:
            return HDDResult(minimal_failing_input=structured_input, test_records=tuple(self._records), nodes=dict(self.nodes))

        self._reduce_node(self.root)

        minimal = self._materialize(self.root, self.active_ids)
        return HDDResult(minimal_failing_input=minimal, test_records=tuple(self._records), nodes=dict(self.nodes))

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

    def _run_test(self, active_ids: Set[int], trial_node_id: Optional[int], trial_depth: Optional[int]) -> bool:
        candidate = self._materialize(self.root, active_ids)
        failed = bool(self.oracle(candidate))
        self._records.append(
            TestRecord(
                test_id=next(self._test_counter),
                active_ids=frozenset(active_ids),
                failed=failed,
                weight=1.0,
                trial_node_id=trial_node_id,
                trial_depth=trial_depth,
            )
        )
        return failed

    def _reduce_node(self, node: Node) -> None:
        children = node.child_nodes()
        for child in children:
            self._reduce_node(child)
        if children:
            self._ddmin_children(children)

    def _ddmin_children(self, children: List[Node]) -> None:
        active_children = [c for c in children if c.id in self.active_ids]
        if len(active_children) <= 1:
            return

        children_by_id = {c.id: c for c in active_children}
        lmin_ids = self._ddmin_reduce(children_by_id)
        lmin_ids = self._ensure_one_minimal(lmin_ids, children_by_id)

        keep = set(lmin_ids)
        for cid in children_by_id:
            if cid not in keep:
                self.active_ids -= self._subtree_ids[cid]

    def _ddmin_reduce(self, children_by_id: Dict[int, Node]) -> List[int]:
        all_ids = list(children_by_id.keys())
        lmin_ids = list(all_ids)
        partitions: List[List[int]] = [list(all_ids)]

        while partitions:
            progressed = False

            for ptn in partitions:
                if self._run_container_test(children_by_id, all_ids, ptn):
                    lmin_ids = list(ptn)
                    partitions = self._even_partition_list([ptn])
                    progressed = True
                    break
            if progressed:
                continue

            for ptn in partitions:
                complement = [i for i in lmin_ids if i not in ptn]
                if complement and self._run_container_test(children_by_id, all_ids, complement):
                    lmin_ids = complement
                    partitions = [p for p in partitions if p is not ptn]
                    progressed = True
                    break
            if progressed:
                continue

            partitions = self._even_partition_list(partitions)

        return lmin_ids

    def _even_partition_list(self, partitions: List[List[int]]) -> List[List[int]]:
        result: List[List[int]] = []
        for ptn in partitions:
            if len(ptn) <= 1:
                continue
            mid = max(1, len(ptn) // 2)
            p1, p2 = ptn[:mid], ptn[mid:]
            if p1:
                result.append(p1)
            if p2:
                result.append(p2)
        return result

    def _ensure_one_minimal(self, lmin_ids: List[int], children_by_id: Dict[int, Node]) -> List[int]:
        all_ids = list(children_by_id.keys())
        changed = True
        while changed:
            changed = False
            for cid in list(lmin_ids):
                candidate = [i for i in lmin_ids if i != cid]
                if not candidate:
                    continue
                if self._run_container_test(children_by_id, all_ids, candidate):
                    lmin_ids = candidate
                    changed = True
                    break
        return lmin_ids

    def _run_container_test(self, children_by_id: Dict[int, Node], all_ids: List[int], keep_ids: List[int]) -> bool:
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
        return self._run_test(trial_active, trial_node_id=trial_node_id, trial_depth=trial_depth)


def run_hdd(structured_input: Any, oracle: Callable[[Any], bool]) -> HDDResult:
    debugger = HierarchicalDeltaDebugger(oracle=oracle)
    return debugger.reduce(structured_input)


def empty_oracle(_: Any) -> bool:
    raise NotImplementedError("Replace empty_oracle with a real test oracle.")


if __name__ == "__main__":
    sample_input = {"a": [1, 2, {"x": "ok", "y": "BUG"}], "b": [3, 4, 5], "c": {"nested": ["fine", "also fine"]}}

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

    debugger = HierarchicalDeltaDebugger(oracle=oracle)
    result = debugger.reduce(sample_input)
    print("Baseline minimal:", result.minimal_failing_input)
    print("Returned tests:", len(result.test_records))
