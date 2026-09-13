"""BugsInPy youtube-dl #2 differential semantic oracle.

The reducible input is an MPEG-DASH XML manifest.  The buggy implementation
merges two formats whose Representation ``id`` values happen to be equal;
the fixed implementation retains both formats.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
BUGS_WORKSPACE_ROOT = REPO_ROOT.parent / "BugsInPy" / "bugs_workspace"
CONTAINER_WORKSPACE_ROOT = Path("/workspace")

BUGGY_ROOT = BUGS_WORKSPACE_ROOT / "youtube-dl_2_buggy" / "youtube-dl"
FIXED_ROOT = BUGS_WORKSPACE_ROOT / "youtube-dl_2_fixed" / "youtube-dl"
SEED_INPUT = BUGGY_ROOT / "test/testdata/mpd/float_duration.mpd"


@dataclass(frozen=True)
class EvaluationResult:
    outcome: str
    buggy_output: dict
    fixed_output: dict
    coverage: set[tuple[str, int]]
    candidate_size: int
    structurally_valid: bool = True
    semantically_valid: bool = True
    runtime_ms: float = 0.0


class YoutubeDL2Adapter:
    def __init__(
        self,
        buggy_root: Path = BUGGY_ROOT,
        fixed_root: Path = FIXED_ROOT,
        timeout: int = 30,
        collect_coverage: bool = True,
    ) -> None:
        self.buggy_root = Path(buggy_root)
        self.fixed_root = Path(fixed_root)
        self.workspace_root = self._detect_workspace_root(self.buggy_root, self.fixed_root)
        self.timeout = timeout
        self.collect_coverage = collect_coverage
        self._validate_paths()

    def evaluate(self, candidate_xml: str) -> EvaluationResult:
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="hddloc_youtubedl2_", dir=str(self.workspace_root)) as temp:
            temp_dir = Path(temp)
            candidate_file = temp_dir / "candidate.mpd"
            candidate_file.write_text(candidate_xml, encoding="utf-8")
            runner_file = temp_dir / "runner.py"
            runner_file.write_text(self._runner_source(), encoding="utf-8")
            coverage_file = temp_dir / ".coverage"

            buggy = self._run_version(
                self.buggy_root,
                candidate_file,
                runner_file,
                coverage_file,
                self.collect_coverage,
            )
            fixed = self._run_version(
                self.fixed_root,
                candidate_file,
                runner_file,
                None,
                False,
            )
            outcome, structurally_valid, semantically_valid = self._classify_outcome(
                buggy, fixed
            )
            coverage = set()
            if self.collect_coverage and coverage_file.exists():
                coverage = self._read_coverage(coverage_file, temp_dir)

            return EvaluationResult(
                outcome=outcome,
                buggy_output=buggy,
                fixed_output=fixed,
                coverage=coverage,
                candidate_size=len(candidate_xml),
                structurally_valid=structurally_valid,
                semantically_valid=semantically_valid,
                runtime_ms=(time.perf_counter() - started) * 1000.0,
            )

    def oracle(self, candidate_xml: str) -> bool:
        return self.evaluate(candidate_xml).outcome == "FAIL"

    @staticmethod
    def _classify_outcome(buggy: dict, fixed: dict) -> tuple[str, bool, bool]:
        if buggy.get("kind") == "formats" and fixed.get("kind") == "formats":
            if buggy.get("output") != fixed.get("output"):
                return "FAIL", True, True
            return "PASS", True, True
        if buggy.get("kind") == "exception" and fixed.get("kind") == "exception":
            return "INVALID", True, False
        # One revision accepting a valid XML candidate while the other rejects
        # it is itself an observable semantic difference.
        return "FAIL", True, True

    def _run_version(
        self,
        project_root: Path,
        candidate_file: Path,
        runner_file: Path,
        coverage_file: Path | None,
        use_coverage: bool,
    ) -> dict:
        container_project_root = self._container_path(project_root)
        container_candidate = self._container_path(candidate_file)
        container_runner = self._container_path(runner_file)
        container_work_dir = self._container_path(candidate_file.parent)
        if use_coverage:
            if coverage_file is None:
                raise ValueError("coverage_file is required when use_coverage=True")
            container_coverage = self._container_path(coverage_file)
            python_command = (
                f"export COVERAGE_FILE={shlex.quote(str(container_coverage))}; "
                f"python3 -m coverage run --source=youtube_dl "
                f"{shlex.quote(str(container_runner))} "
                f"{shlex.quote(str(container_project_root))} "
                f"{shlex.quote(str(container_candidate))}"
            )
        else:
            python_command = (
                f"python3 {shlex.quote(str(container_runner))} "
                f"{shlex.quote(str(container_project_root))} "
                f"{shlex.quote(str(container_candidate))}"
            )

        command = [
            "docker",
            "exec",
            "bugsinpy_bg",
            "bash",
            "-c",
            (
                "export PATH=$PATH:/BugsInPy/framework/bin:/framework/bin; "
                f"cd {shlex.quote(str(container_work_dir))} && {python_command}"
            ),
        ]
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout,
            )
        except FileNotFoundError as exc:
            return {"kind": "exception", "exception_type": type(exc).__name__, "message": str(exc)}
        except subprocess.TimeoutExpired:
            return {"kind": "exception", "exception_type": "TimeoutExpired", "message": "Execution timed out"}

        try:
            return json.loads(result.stdout.strip())
        except json.JSONDecodeError:
            return {
                "kind": "exception",
                "exception_type": "RunnerFailure",
                "message": (
                    f"returncode={result.returncode}; stdout={result.stdout!r}; "
                    f"stderr={result.stderr!r}"
                ),
            }

    def _read_coverage(self, coverage_file: Path, temp_dir: Path) -> set[tuple[str, int]]:
        json_file = temp_dir / "coverage.json"
        command = [
            "docker",
            "exec",
            "bugsinpy_bg",
            "bash",
            "-c",
            (
                f"export COVERAGE_FILE={shlex.quote(str(self._container_path(coverage_file)))}; "
                f"python3 -m coverage json -o {shlex.quote(str(self._container_path(json_file)))}"
            ),
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0 or not json_file.exists():
            return set()
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return set()

        executed: set[tuple[str, int]] = set()
        buggy_root = self.buggy_root.resolve()
        for filename, information in data.get("files", {}).items():
            path = Path(filename)
            try:
                host_path = self.workspace_root / path.relative_to(CONTAINER_WORKSPACE_ROOT)
            except ValueError:
                host_path = path
            try:
                relative = host_path.relative_to(buggy_root)
            except ValueError:
                continue
            if "test" in relative.parts or "tests" in relative.parts:
                continue
            for line in information.get("executed_lines", []):
                executed.add((str(relative), int(line)))
        return executed

    @staticmethod
    def _runner_source() -> str:
        return r'''
import json
import sys
from pathlib import Path

project_root = Path(sys.argv[1]).resolve()
candidate_file = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(project_root))

try:
    from youtube_dl.compat import compat_etree_fromstring
    from youtube_dl.extractor.common import InfoExtractor

    class LocalDownloader:
        params = {}

        def report_warning(self, message):
            pass

        def to_screen(self, message):
            pass

    document = compat_etree_fromstring(candidate_file.read_bytes())
    formats = InfoExtractor(LocalDownloader())._parse_mpd_formats(
        document,
        mpd_url='http://unknown/manifest.mpd',
    )
    print(json.dumps({'kind': 'formats', 'output': formats}, sort_keys=True))
except Exception as exc:
    print(json.dumps({
        'kind': 'exception',
        'exception_type': type(exc).__name__,
        'message': str(exc),
    }, sort_keys=True))
'''

    def _validate_paths(self) -> None:
        if not self.buggy_root.exists():
            raise FileNotFoundError(f"Buggy checkout missing: {self.buggy_root}")
        if not self.fixed_root.exists():
            raise FileNotFoundError(f"Fixed checkout missing: {self.fixed_root}")

    @staticmethod
    def _detect_workspace_root(buggy_root: Path, fixed_root: Path) -> Path:
        for project_root in (buggy_root, fixed_root):
            resolved = project_root.resolve()
            for ancestor in (resolved, *resolved.parents):
                if ancestor.name == "bugs_workspace":
                    return ancestor
        return Path(os.path.commonpath([str(buggy_root.resolve()), str(fixed_root.resolve())]))

    def _container_path(self, host_path: Path) -> Path:
        relative = Path(host_path).resolve().relative_to(self.workspace_root.resolve())
        return CONTAINER_WORKSPACE_ROOT / relative
