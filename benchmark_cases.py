"""Declarative registry for the selected HDD-LOC BugsInPy cases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from bugsinpy_case_adapter import BUGS_WORKSPACE_ROOT, BugsInPyCaseAdapter
from execution_records import CandidateEvaluation, Outcome, SourceLocation
from structured_markup import StructuredMarkup
from youtube_dl2_adapter import SEED_INPUT as YOUTUBE_DL2_SEED
from youtube_dl2_adapter import YoutubeDL2Adapter
from black2_adapter import Black2Adapter


REPO_ROOT = Path(__file__).resolve().parent
BUGSINPY_ROOT = REPO_ROOT.parent / "BugsInPy"


def serialize_code_tree(candidate: Any) -> str:
    """Serialize ordered code constructs while preserving complete units."""

    if isinstance(candidate, str):
        return candidate
    if isinstance(candidate, dict):
        return "".join(serialize_code_tree(value) for value in candidate.values())
    if isinstance(candidate, (list, tuple)):
        return "".join(serialize_code_tree(value) for value in candidate)
    raise ValueError(f"unsupported code-tree value: {type(candidate).__name__}")


@dataclass(frozen=True)
class CaseRuntime:
    structured_input: Any
    serializer: Callable[[Any], str]
    evaluator: Callable[[Any], CandidateEvaluation]
    serialized_evaluator: Callable[[str], CandidateEvaluation]


@dataclass(frozen=True)
class BenchmarkCase:
    key: str
    project: str
    bug_id: int
    primary_module: str
    fault_category: str
    input_kind: str
    coverage_source: str
    seed: Any = None
    adapter_kind: str = "generic"

    @property
    def buggy_root(self) -> Path:
        return BUGS_WORKSPACE_ROOT / f"{self.key}_buggy" / self.project

    @property
    def fixed_root(self) -> Path:
        return BUGS_WORKSPACE_ROOT / f"{self.key}_fixed" / self.project

    @property
    def patch_path(self) -> Path:
        return BUGSINPY_ROOT / "projects" / self.project / "bugs" / str(self.bug_id) / "bug_patch.txt"

    def build_runtime(self, *, timeout: int = 30) -> CaseRuntime:
        if self.adapter_kind == "black2":
            adapter = Black2Adapter(
                buggy_root=self.buggy_root,
                fixed_root=self.fixed_root,
                timeout=timeout,
                collect_coverage=True,
            )

            def evaluate(candidate: Any) -> CandidateEvaluation:
                result = adapter.evaluate(serialize_code_tree(candidate))
                return _candidate_evaluation(result)

            def evaluate_serialized(candidate: str) -> CandidateEvaluation:
                return _candidate_evaluation(adapter.evaluate(candidate))

            return CaseRuntime(
                self.seed, serialize_code_tree, evaluate, evaluate_serialized
            )

        if self.adapter_kind == "youtube_dl2":
            markup = StructuredMarkup.parse_xml(YOUTUBE_DL2_SEED.read_text(encoding="utf-8"))
            adapter = YoutubeDL2Adapter(
                buggy_root=self.buggy_root,
                fixed_root=self.fixed_root,
                timeout=timeout,
                collect_coverage=True,
            )

            def evaluate(candidate: Any) -> CandidateEvaluation:
                try:
                    source = markup.serialize(candidate)
                except ValueError:
                    return CandidateEvaluation(
                        outcome=Outcome.INVALID,
                        structurally_valid=False,
                        semantically_valid=False,
                    )
                return _candidate_evaluation(adapter.evaluate(source))

            def evaluate_serialized(candidate: str) -> CandidateEvaluation:
                return _candidate_evaluation(adapter.evaluate(candidate))

            return CaseRuntime(
                markup.tree, markup.serialize, evaluate, evaluate_serialized
            )

        adapter = BugsInPyCaseAdapter(
            self.key,
            self.buggy_root,
            self.fixed_root,
            self.coverage_source,
            timeout=timeout,
            collect_coverage=True,
        )
        def evaluate(candidate: Any) -> CandidateEvaluation:
            return _candidate_evaluation(adapter.evaluate(candidate))

        def evaluate_serialized(candidate: str) -> CandidateEvaluation:
            import json

            try:
                structured_candidate = json.loads(candidate)
            except (json.JSONDecodeError, TypeError):
                return CandidateEvaluation(
                    outcome=Outcome.INVALID,
                    structurally_valid=False,
                    semantically_valid=False,
                )
            return evaluate(structured_candidate)

        return CaseRuntime(
            self.seed,
            _json_serializer,
            evaluate,
            evaluate_serialized,
        )


def _json_serializer(candidate: Any) -> str:
    import json

    return json.dumps(candidate, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _candidate_evaluation(result: Any) -> CandidateEvaluation:
    return CandidateEvaluation(
        outcome=Outcome(result.outcome),
        coverage=frozenset(
            SourceLocation(filename, line) for filename, line in result.coverage
        ),
        structurally_valid=result.structurally_valid,
        semantically_valid=result.semantically_valid,
        runtime_ms=result.runtime_ms,
    )


BLACK2_TREE = {
    "document": {
        "first_example": {
            "fmt_off": "# fmt: off\n",
            "decorator": "@test([\n    1, 2,\n    3, 4,\n])\n",
            "fmt_on": "# fmt: on\n",
            "function": "def f(): pass\n\n",
        },
        "second_example": {
            "decorator": "@test([\n    1, 2,\n    3, 4,\n])\n",
            "function": "def f(): pass\n",
        },
    }
}


SELECTED_CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        "black_1", "black", 1, "black.py", "multiprocessing fallback",
        "directory of Python source files", "black",
        {"files": ["one.py", "two.py"], "environment": {"multiprocessing": False}},
    ),
    BenchmarkCase(
        "black_2", "black", 2, "black.py", "format-toggle state logic",
        "Python concrete-syntax construct tree", "black", BLACK2_TREE, "black2",
    ),
    BenchmarkCase(
        "black_3", "black", 3, "black.py", "CLI path validation",
        "CLI/config scenario tree", "black",
        {
            "source": "x = 1\n",
            "missing_config_name": "missing.toml",
            "arguments": {"use_config": True},
        },
    ),
    BenchmarkCase(
        "tornado_1", "tornado", 1, "tornado/websocket.py", "WebSocket nodelay delegation",
        "WebSocket action/state tree", "tornado",
        {"value": True, "connection": {"established": True}},
    ),
    BenchmarkCase(
        "tornado_2", "tornado", 2, "tornado/http1connection.py", "HTTP chunking boundary",
        "HTTP request tree", "tornado",
        {"method": "PUT", "path": "/start", "headers": {"Transfer-Encoding": "chunked"}},
    ),
    BenchmarkCase(
        "tornado_3", "tornado", 3, "tornado/httpclient.py", "client cache lifecycle",
        "state/action-sequence tree", "tornado",
        {"cache_state": "missing", "actions": ["create", "clear_weakref", "close"]},
    ),
    BenchmarkCase(
        "tornado_4", "tornado", 4, "tornado/web.py", "HTTP range validation",
        "HTTP Range-header tree", "tornado",
        {"range": {"kind": "explicit", "start": 10, "end": 3}, "resource_size": 26},
    ),
    BenchmarkCase(
        "youtube-dl_1", "youtube-dl", 1, "youtube_dl/utils.py", "metadata filter semantics",
        "filter-expression and metadata tree", "youtube_dl",
        {
            "clauses": [
                {"key": "x", "op": ">", "value": 1},
                {"key": "title"},
                {"key": "is_live"},
            ],
            "metadata": {"x": 1200, "title": "abc", "is_live": False},
        },
    ),
    BenchmarkCase(
        "youtube-dl_2", "youtube-dl", 2, "youtube_dl/extractor/common.py", "duplicate MPD representation IDs",
        "MPEG-DASH XML element tree", "youtube_dl", adapter_kind="youtube_dl2",
    ),
    BenchmarkCase(
        "tqdm_1", "tqdm", 1, "tqdm/contrib/__init__.py", "enumeration start semantics",
        "iteration-options tree", "tqdm",
        {"iterable": [0, 1, 2], "start": 42},
    ),
    BenchmarkCase(
        "tqdm_2", "tqdm", 2, "tqdm/std.py;tqdm/utils.py", "ANSI display trimming",
        "formatting-options tree", "tqdm",
        {
            "n": 0,
            "total": 1000,
            "elapsed": 13,
            "options": {
                "ncols": 10,
                "bar_format": "*****\u001b[22m*****\u001b[0m**{bar:10}$$$$$$$$$$",
            },
        },
    ),
    BenchmarkCase(
        "luigi_1", "luigi", 1, "luigi/server.py", "metrics-handler delegation",
        "metrics response/state tree", "luigi",
        {"metrics_available": True, "request": {"method": "GET", "format": "prometheus"}},
    ),
)


CASES_BY_KEY = {case.key: case for case in SELECTED_CASES}
