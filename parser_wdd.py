"""Parser-preserving, level-based weighted delta debugging for Python source.

This reducer is used by HDD-LOC.  Reduction units are Python AST statements
that occur in grammar-sanctioned statement lists.  Candidates are materialized
by deleting their original source spans, so comments/formatting/pragmas are
preserved as far as possible.  Every candidate is reparsed before the semantic
oracle is called.

The WDD loop follows the supplied weighted-delta-debugging pseudocode:
    * start from one partition containing the current level,
    * try partitions directly,
    * try complements,
    * repartition by approximately equal token weight,
    * finish with an ensureOneMinimal pass.

HDD remains breadth-first over parser-tree statement levels.  A final global
ensureOneMinimal pass is also performed after all levels, because a reduction
at a deeper level can make an ancestor statement removable.
"""

from __future__ import annotations

import ast
import io
import itertools
import tokenize
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from hdd_algorithm import HDDResult, Node, TestRecord


@dataclass(frozen=True)
class _Element:
    id: int
    node: ast.stmt
    depth: int
    path: Tuple[str, ...]
    parent_element_id: Optional[int]


class PythonParserWeightedDeltaDebugger:
    """Grammar-preserving, level-based weighted HDD/WDD reducer for Python."""

    def __init__(self, oracle: Callable[[str], bool]) -> None:
        self.oracle = oracle
        self.root: Optional[Node] = None
        self.nodes: Dict[int, Node] = {}
        self.active_ids: Set[int] = set()
        self._tree: Optional[ast.Module] = None
        self._source = ""
        self._elements: Dict[int, _Element] = {}
        self._levels: Dict[int, List[int]] = {}
        self._records: List[TestRecord] = []
        self._counter = itertools.count()
        self._test_counter = itertools.count()
        self._weight_cache: Dict[int, int] = {}
        self._line_char_offsets: List[int] = []
        self._line_byte_to_char: List[Dict[int, int]] = []

    def reduce(self, source: str) -> HDDResult:
        """Reduce ``source`` while preserving the oracle result and valid syntax."""
        self._source = source
        self._tree = ast.parse(source)
        self._elements = {}
        self._levels = {}
        self.nodes = {}
        self._records = []
        self._counter = itertools.count()
        self._test_counter = itertools.count()
        self._weight_cache = {}
        self._build_source_offsets(source)

        self._index_statement_lists(
            self._tree,
            depth=1,
            path=("module",),
            parent_element_id=None,
        )

        self.root = Node(id=-1, path=(), depth=0, kind="python_module")
        self.nodes[-1] = self.root
        self.active_ids = set(self._elements)

        # Preserve the original behavior: reduction only starts when the
        # complete source satisfies the semantic property.
        if not self._run_test(
            self.active_ids,
            None,
            None,
            self._candidate_weight(self.active_ids),
        ):
            return HDDResult(source, tuple(self._records), dict(self.nodes))

        # HDD is breadth-first over statement nesting levels.
        for depth in sorted(self._levels):
            level = [
                element_id
                for element_id in self._levels[depth]
                if self._is_visible(element_id)
            ]
            if level:
                self._wddmin_level(level, depth)

        # A deeper reduction can make an ancestor removable.  Therefore the
        # WDD-level minimality pass is followed by one final global pass.
        self._ensure_global_one_minimal()

        return HDDResult(
            self._materialize(self.root, self.active_ids),
            tuple(self._records),
            dict(self.nodes),
        )

    # ------------------------------------------------------------------
    # AST indexing
    # ------------------------------------------------------------------

    def _index_statement_lists(
        self,
        owner: ast.AST,
        depth: int,
        path: Tuple[str, ...],
        parent_element_id: Optional[int],
    ) -> None:
        """Index statements in every AST statement-list field.

        The previous implementation only traversed lists where every item was
        ``ast.stmt``.  That misses containers such as ``Try.handlers``
        (``ExceptHandler``) and ``Match.cases`` (``match_case``).  Here we
        recurse through *all* AST-containing lists, while only making actual
        ``ast.stmt`` nodes reduction elements.
        """
        for field, value in ast.iter_fields(owner):
            field_path = path + (field,)

            if isinstance(value, list):
                for index, item in enumerate(value):
                    item_path = field_path + (str(index),)

                    if isinstance(item, ast.stmt):
                        element_id = next(self._counter)
                        setattr(item, "_hddloc_id", element_id)
                        element_path = item_path + (type(item).__name__,)
                        element = _Element(
                            id=element_id,
                            node=item,
                            depth=depth,
                            path=element_path,
                            parent_element_id=parent_element_id,
                        )
                        self._elements[element_id] = element
                        self._levels.setdefault(depth, []).append(element_id)
                        self.nodes[element_id] = Node(
                            id=element_id,
                            path=element_path,
                            depth=depth,
                            kind="python_statement",
                            value=type(item).__name__,
                        )
                        self._index_statement_lists(
                            item,
                            depth=depth + 1,
                            path=element_path,
                            parent_element_id=element_id,
                        )
                    elif isinstance(item, ast.AST):
                        # Non-statement AST containers (e.g. ExceptHandler,
                        # match_case) do not create a reduction element, but
                        # their nested statement suites still belong one level
                        # below the surrounding statement.
                        self._index_statement_lists(
                            item,
                            depth=depth,
                            path=item_path,
                            parent_element_id=parent_element_id,
                        )

            elif isinstance(value, ast.AST):
                # Scalar AST wrappers (e.g. decorators/expressions) are
                # traversed so statement lists nested below them are not lost.
                self._index_statement_lists(
                    value,
                    depth=depth,
                    path=field_path,
                    parent_element_id=parent_element_id,
                )

    def _is_visible(self, element_id: int) -> bool:
        """Return whether an active element is not hidden by a removed ancestor."""
        current = self._elements[element_id]
        while current.parent_element_id is not None:
            parent_id = current.parent_element_id
            if parent_id not in self.active_ids:
                return False
            current = self._elements[parent_id]
        return element_id in self.active_ids

    # ------------------------------------------------------------------
    # Weighted delta debugging at one HDD level
    # ------------------------------------------------------------------

    def _wddmin_level(self, level_ids: List[int], depth: int) -> None:
        """Apply the supplied WDD procedure to one HDD tree level."""
        best = list(level_ids)

        # Match the supplied WDD pseudocode: begin with the entire current
        # list as one partition, then recursively split after testing it.
        partitions: List[List[int]] = [list(best)]

        while partitions:
            reduced = False

            # 1. Test partitions directly.
            for partition in partitions:
                if self._test_level(level_ids, partition, depth):
                    best = list(partition)
                    partitions = self._weighted_partition([best])
                    reduced = True
                    break

            if reduced:
                continue

            # 2. Test complements.
            deleted_partition = False
            for partition in list(partitions):
                complement = [
                    element_id
                    for element_id in best
                    if element_id not in partition
                ]
                if self._test_level(level_ids, complement, depth):
                    best = complement
                    # Remove this exact partition by equality rather than
                    # object identity so equivalent reconstructed partitions
                    # cannot cause surprising behavior.
                    partitions.remove(partition)
                    deleted_partition = True
                    break

            if deleted_partition:
                continue

            # 3. Repartition.  weighted_partition() naturally returns an empty
            # list once every partition is a singleton.
            partitions = self._weighted_partition(partitions)

        # 4. WDD's one-element minimization for this level.
        best_set = set(best)
        changed = True
        while changed:
            changed = False
            for element_id in list(best):
                candidate = [
                    item for item in best
                    if item != element_id
                ]
                if self._test_level(level_ids, candidate, depth):
                    best = candidate
                    best_set = set(best)
                    changed = True
                    break

        # Deactivate only elements at this level.  Descendants remain selected
        # internally so that if the ancestor remains active their original
        # content is restored automatically.
        self.active_ids.difference_update(
            set(level_ids).difference(best_set)
        )

    def _ensure_global_one_minimal(self) -> None:
        """Finish with a global one-element-minimality pass.

        This is deliberately performed after all HDD levels.  Removing a deep
        statement can make a statement at an earlier level unnecessary, so a
        per-level pass alone cannot guarantee a globally 1-minimal result.
        """
        changed = True
        while changed:
            changed = False

            visible_active = [
                element_id
                for element_id in self._elements
                if element_id in self.active_ids and self._is_visible(element_id)
            ]

            for element_id in visible_active:
                candidate_active = set(self.active_ids)
                candidate_active.remove(element_id)
                weight = self._candidate_weight(candidate_active)

                if self._run_test(
                    candidate_active,
                    element_id,
                    self._elements[element_id].depth,
                    weight,
                ):
                    self.active_ids = candidate_active
                    changed = True
                    break

    def _test_level(
        self,
        level_ids: Iterable[int],
        keep_ids: Iterable[int],
        depth: int,
    ) -> bool:
        level_set = set(level_ids)
        keep = set(keep_ids)
        trial_active = (self.active_ids - level_set) | keep
        weight = self._candidate_weight(trial_active)
        trial_node_id = next(iter(keep)) if len(keep) == 1 else None
        return self._run_test(
            trial_active,
            trial_node_id,
            depth,
            weight,
        )

    # ------------------------------------------------------------------
    # Weighted partitioning
    # ------------------------------------------------------------------

    def _weighted_partition(self, partitions: List[List[int]]) -> List[List[int]]:
        """Split each partition into two contiguous groups near half weight.

        This is a deterministic source-order split.  It matches the intended
        WDD heuristic while avoiding an exponential arbitrary-subset search.
        """
        result: List[List[int]] = []

        for partition in partitions:
            if len(partition) <= 1:
                continue

            weights = [
                self._token_weight(element_id)
                for element_id in partition
            ]
            expected = 0.5 * sum(weights)

            cumulative = 0
            split_at = 1
            closest = float("inf")

            for index in range(1, len(weights)):
                cumulative += weights[index - 1]
                distance = abs(cumulative - expected)
                if distance < closest:
                    closest = distance
                    split_at = index

            left = partition[:split_at]
            right = partition[split_at:]
            if left and right:
                result.extend((left, right))

        return result

    def _token_weight(self, element_id: int) -> int:
        """Return a stable positive token weight for an AST statement."""
        cached = self._weight_cache.get(element_id)
        if cached is not None:
            return cached

        text = ast.unparse(self._elements[element_id].node)
        weight = sum(
            1
            for token in tokenize.generate_tokens(
                io.StringIO(text).readline
            )
            if token.type
            not in (
                tokenize.ENCODING,
                tokenize.ENDMARKER,
                tokenize.NEWLINE,
                tokenize.NL,
                tokenize.INDENT,
                tokenize.DEDENT,
            )
        )
        weight = max(1, weight)
        self._weight_cache[element_id] = weight
        return weight

    def _candidate_weight(self, active_ids: Iterable[int]) -> float:
        return float(sum(self._token_weight(element_id) for element_id in active_ids))

    # ------------------------------------------------------------------
    # Oracle / candidate materialization
    # ------------------------------------------------------------------

    def _run_test(
        self,
        active_ids: Set[int],
        trial_node_id: Optional[int],
        depth: Optional[int],
        weight: float,
    ) -> bool:
        candidate = self._materialize(self.root, active_ids)

        # Invalid candidates are rejected before reaching the semantic oracle.
        # This makes grammar-invalid proposals behave like ordinary failed
        # reductions instead of crashing the entire WDD run.
        try:
            ast.parse(candidate)
        except SyntaxError:
            failed = False
            self._records.append(
                TestRecord(
                    next(self._test_counter),
                    frozenset(active_ids),
                    failed,
                    1.0 / weight if weight else 1.0,
                    trial_node_id,
                    depth,
                )
            )
            return False

        failed = bool(self.oracle(candidate))
        self._records.append(
            TestRecord(
                next(self._test_counter),
                frozenset(active_ids),
                failed,
                1.0 / weight if weight else 1.0,
                trial_node_id,
                depth,
            )
        )
        return failed

    def _build_source_offsets(self, source: str) -> None:
        """Build UTF-8-byte-column -> Python-character offsets per source line.

        Python AST ``col_offset`` / ``end_col_offset`` values are UTF-8 byte
        offsets, while Python ``str`` slicing uses Unicode code-point indexes.
        The mapping prevents source corruption when non-ASCII characters occur
        before a reduction span.
        """
        self._line_char_offsets = [0]
        self._line_byte_to_char = []

        char_position = 0
        lines = source.splitlines(keepends=True)
        for line in lines:
            byte_to_char: Dict[int, int] = {0: 0}
            byte_position = 0
            char_in_line = 0

            for char in line:
                encoded_len = len(char.encode("utf-8"))
                byte_position += encoded_len
                char_in_line += 1
                byte_to_char[byte_position] = char_in_line

            self._line_byte_to_char.append(byte_to_char)
            char_position += len(line)
            self._line_char_offsets.append(char_position)

    def _offset(self, line: int, byte_column: int) -> int:
        if line < 1:
            raise ValueError(f"invalid AST line number: {line}")

        line_index = line - 1
        if line_index >= len(self._line_char_offsets) - 1:
            # Defensive fallback for unusual source endings.
            return len(self._source)

        byte_to_char = self._line_byte_to_char[line_index]
        if byte_column in byte_to_char:
            char_column = byte_to_char[byte_column]
        else:
            # AST positions should land on UTF-8 character boundaries.  For
            # extra robustness, decode the prefix up to the requested byte.
            lines = self._source.splitlines(keepends=True)
            raw_line = lines[line_index].encode("utf-8")
            prefix = raw_line[:byte_column]
            char_column = len(prefix.decode("utf-8"))

        return self._line_char_offsets[line_index] + char_column

    def _materialize(
        self,
        _root: Optional[Node],
        active_ids: Set[int],
    ) -> str:
        if self._tree is None:
            raise ValueError("no parsed source -- call reduce() first")

        edits: List[Tuple[int, int, str]] = []

        # Delete original source spans for inactive statements.  If an ancestor
        # is already inactive, its span covers the descendants, so descendants
        # do not add overlapping deletions.
        for element in self._elements.values():
            if element.id in active_ids:
                continue

            parent_id = element.parent_element_id
            if parent_id is not None and parent_id not in active_ids:
                continue

            node = element.node
            start = self._offset(node.lineno, node.col_offset)
            end = self._offset(
                getattr(node, "end_lineno", node.lineno),
                getattr(node, "end_col_offset", node.col_offset),
            )
            edits.append((start, end, ""))

        # Insert ``pass`` into every originally non-empty AST statement-list
        # suite whose children have all been removed.  This handles not just the
        # common .body fields, but also else/finally suites, ExceptHandler.body,
        # and match_case.body.  Module.body is intentionally excluded because an
        # empty module is valid Python.
        for owner in ast.walk(self._tree):
            if isinstance(owner, ast.Module):
                continue

            owner_id = getattr(owner, "_hddloc_id", None)
            if owner_id is not None and owner_id not in active_ids:
                continue

            for field, value in ast.iter_fields(owner):
                if not isinstance(value, list) or not value:
                    continue
                if not all(isinstance(item, ast.stmt) for item in value):
                    continue

                statement_ids = [getattr(statement, "_hddloc_id", None) for statement in value]
                if not statement_ids or any(item_id is None for item_id in statement_ids):
                    continue

                if all(item_id not in active_ids for item_id in statement_ids):
                    first = value[0]
                    start = self._offset(first.lineno, first.col_offset)
                    end = self._offset(
                        getattr(first, "end_lineno", first.lineno),
                        getattr(first, "end_col_offset", first.col_offset),
                    )
                    edits.append((start, end, "pass"))

        source = self._source

        # Apply right-to-left so all positions continue to refer to the original
        # source.  For an identical span, a grammar-preserving ``pass`` wins over
        # a deletion.
        selected: Dict[Tuple[int, int], str] = {}
        for start, end, replacement in edits:
            key = (start, end)
            if replacement or key not in selected:
                selected[key] = replacement

        for (start, end), replacement in sorted(
            selected.items(),
            key=lambda item: (item[0][0], item[0][1]),
            reverse=True,
        ):
            source = source[:start] + replacement + source[end:]

        return source
