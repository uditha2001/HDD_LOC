from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
BUGS_WORKSPACE_ROOT = REPO_ROOT.parent / "BugsInPy" / "bugs_workspace"
CONTAINER_WORKSPACE_ROOT = Path("/workspace")

BUGGY_ROOT = BUGS_WORKSPACE_ROOT / "black_2_buggy" / "black"
FIXED_ROOT = BUGS_WORKSPACE_ROOT / "black_2_fixed" / "black"

# Original real BugsInPy structured input.
SEED_INPUT = BUGGY_ROOT / "tests/data/fmtonoff.py"


@dataclass
class EvaluationResult:
    # FAIL = semantic bug is still reproduced.
    # PASS = reduction removed the semantic difference.
    outcome: str

    buggy_output: dict
    fixed_output: dict

    # Lines executed in the buggy program.
    coverage: set[tuple[str, int]]

    candidate_size: int

    structurally_valid: bool = True
    semantically_valid: bool = True
    runtime_ms: float = 0.0


class Black2Adapter:
    def __init__(
        self,
        buggy_root: Path = BUGGY_ROOT,
        fixed_root: Path = FIXED_ROOT,
        timeout: int = 30,
        collect_coverage: bool = True,
    ):
        self.buggy_root = Path(buggy_root)
        self.fixed_root = Path(fixed_root)
        self.workspace_root = self._detect_workspace_root(self.buggy_root, self.fixed_root)
        self.timeout = timeout
        self.collect_coverage = collect_coverage

        self._validate_paths()

    # =========================================================
    # Public API for HDD / HDD-LOC
    # =========================================================

    def evaluate(self, candidate_source: str) -> EvaluationResult:
        """
        Evaluate one HDD-generated candidate.

        FAIL:
            buggy Black and fixed Black behave differently.

        PASS:
            they behave the same.
        """

        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="hddloc_black2_", dir=str(self.workspace_root)) as temp:
            temp_dir = Path(temp)

            candidate_file = temp_dir / "candidate.py"
            candidate_file.write_text(candidate_source, encoding="utf-8")

            runner_file = temp_dir / "runner.py"
            runner_file.write_text(self._runner_source(), encoding="utf-8")

            # ---------------------------------------------
            # Run buggy version + collect coverage
            # ---------------------------------------------

            coverage_file = temp_dir / ".coverage"
            buggy_result = self._run_version(
                project_root=self.buggy_root,
                candidate_file=candidate_file,
                runner_file=runner_file,
                coverage_file=coverage_file,
                use_coverage=self.collect_coverage,
            )

            # ---------------------------------------------
            # Run fixed version
            # ---------------------------------------------

            fixed_result = self._run_version(
                project_root=self.fixed_root,
                candidate_file=candidate_file,
                runner_file=runner_file,
                coverage_file=None,
                use_coverage=False,
            )

            # ---------------------------------------------
            # Differential semantic oracle
            # ---------------------------------------------

            outcome, structurally_valid, semantically_valid = self._classify_outcome(
                buggy_result, fixed_result
            )

            # ---------------------------------------------
            # Read buggy-program coverage
            # ---------------------------------------------

            coverage = set()
            if self.collect_coverage and coverage_file.exists():
                coverage = self._read_coverage(coverage_file, temp_dir)

            return EvaluationResult(
                outcome=outcome,
                buggy_output=buggy_result,
                fixed_output=fixed_result,
                coverage=coverage,
                candidate_size=len(candidate_source),
                structurally_valid=structurally_valid,
                semantically_valid=semantically_valid,
                runtime_ms=(time.perf_counter() - started) * 1000.0,
            )

    def oracle(self, candidate_source: str) -> bool:
        """Boolean oracle compatible with hdd_algorithm.py and hdd_baseline.py."""

        return self.evaluate(candidate_source).outcome == "FAIL"

    def _classify_outcome(self, buggy: dict, fixed: dict) -> tuple[str, bool, bool]:
        """Return FAIL/PASS/INVALID without turning unusable inputs into tests."""

        if buggy["kind"] != fixed["kind"]:
            return "FAIL", True, True

        if buggy["kind"] == "formatted":
            return (
                ("FAIL", True, True)
                if buggy["output"] != fixed["output"]
                else ("PASS", True, True)
            )

        if buggy["kind"] == "exception":
            exceptions_differ = (
                buggy["exception_type"] != fixed["exception_type"]
                or buggy["message"] != fixed["message"]
            )
            if exceptions_differ:
                return "FAIL", True, True

            structural_exception_names = {
                "IndentationError",
                "InvalidInput",
                "ParseError",
                "SyntaxError",
                "TokenError",
            }
            structurally_valid = buggy["exception_type"] not in structural_exception_names
            return "INVALID", structurally_valid, False

        return (
            ("FAIL", True, True)
            if buggy != fixed
            else ("PASS", True, True)
        )

    # =========================================================
    # Differential oracle
    # =========================================================

    def _bug_is_preserved(self, buggy: dict, fixed: dict) -> bool:
        """
        Black #2 semantic oracle.

        Candidate is interesting when the buggy and fixed
        versions produce different observable formatting
        behaviour.
        """

        outcome, _, _ = self._classify_outcome(buggy, fixed)
        return outcome == "FAIL"

    # =========================================================
    # Execute one Black version
    # =========================================================

    def _run_version(
        self,
        project_root: Path,
        candidate_file: Path,
        runner_file: Path,
        coverage_file: Path | None,
        use_coverage: bool,
    ) -> dict:
        container_project_root = self._container_path(project_root)
        container_candidate_file = self._container_path(candidate_file)
        container_runner_file = self._container_path(runner_file)
        container_work_dir = self._container_path(candidate_file.parent)

        if use_coverage:
            if coverage_file is None:
                raise ValueError("coverage_file is required when use_coverage=True")
            container_coverage_file = self._container_path(coverage_file)
            # Added --source to force tracking of black source files + switched to python3
            python_command = (
                f"export COVERAGE_FILE={shlex.quote(str(container_coverage_file))}; "
                f"python3 -m coverage run --source=black "
                f"{shlex.quote(str(container_runner_file))} "
                f"{shlex.quote(str(container_project_root))} "
                f"{shlex.quote(str(container_candidate_file))}"
            )
        else:
            python_command = (
                f"python3 "
                f"{shlex.quote(str(container_runner_file))} "
                f"{shlex.quote(str(container_project_root))} "
                f"{shlex.quote(str(container_candidate_file))}"
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
            return {
                "kind": "exception",
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }
        except subprocess.TimeoutExpired:
            return {
                "kind": "exception",
                "exception_type": "TimeoutExpired",
                "message": "Execution timed out",
            }

        # runner.py prints a single JSON string - parse it into a dict
        output = result.stdout.strip()

        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return {
                "kind": "exception",
                "exception_type": "RunnerFailure",
                "message": (
                    f"returncode={result.returncode}; "
                    f"stdout={result.stdout!r}; "
                    f"stderr={result.stderr!r}"
                ),
            }
    # =========================================================
    # Coverage extraction
    # =========================================================

    def _read_coverage(self, coverage_file: Path, temp_dir: Path) -> set[tuple[str, int]]:
        json_file = temp_dir / "coverage.json"
        container_coverage_file = self._container_path(coverage_file)
        container_json_file = self._container_path(json_file)

        # Generate JSON inside Docker container where coverage is installed
        cmd = [
            "docker",
            "exec",
            "bugsinpy_bg",
            "bash",
            "-c",
            (
                f"export COVERAGE_FILE={shlex.quote(str(container_coverage_file))}; "
                f"python3 -m coverage json -o {shlex.quote(str(container_json_file))}"
            ),
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0 or not json_file.exists():
            return set()

        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception:
            return set()

        executed = set()
        buggy_root = self.buggy_root.resolve()

        for filename, information in data.get("files", {}).items():
            path = Path(filename)

            # Map container path (/workspace/...) to host path
            try:
                rel_workspace = path.relative_to(CONTAINER_WORKSPACE_ROOT)
                host_path = self.workspace_root / rel_workspace
            except ValueError:
                host_path = path

            # Filter to buggy Black files
            try:
                relative = host_path.relative_to(buggy_root)
            except ValueError:
                continue

            if "tests" in relative.parts:
                continue

            for line in information.get("executed_lines", []):
                executed.add((str(relative), int(line)))

        return executed
    # =========================================================
    # Child process code
    # =========================================================

    @staticmethod
    def _runner_source() -> str:
        return r'''
import json
import sys
from pathlib import Path


project_root = Path(sys.argv[1]).resolve()
candidate_file = Path(sys.argv[2]).resolve()

# Force import from the requested BugsInPy checkout.
sys.path.insert(0, str(project_root))

try:
    import black
except Exception as exc:
    print(json.dumps({
        "kind": "exception",
        "exception_type": type(exc).__name__,
        "message": "Black import failed: " + str(exc),
    }))
    sys.exit(0)


try:
    source = candidate_file.read_text(encoding="utf-8")

    try:
        result = black.format_file_contents(
            src_contents=source,
            fast=True,
            mode=black.FileMode(),
        )

        print(json.dumps({
            "kind": "formatted",
            "output": result,
        }))

    except black.NothingChanged:
        # NothingChanged is normal Black behaviour.
        print(json.dumps({
            "kind": "formatted",
            "output": source,
        }))

except Exception as exc:
    print(json.dumps({
        "kind": "exception",
        "exception_type": type(exc).__name__,
        "message": str(exc),
    }))
'''

    # =========================================================
    # Validation
    # =========================================================

    def _validate_paths(self):
        if not self.buggy_root.exists():
            raise FileNotFoundError(f"Buggy checkout missing: {self.buggy_root}")

        if not self.fixed_root.exists():
            raise FileNotFoundError(f"Fixed checkout missing: {self.fixed_root}")

    def _detect_workspace_root(self, buggy_root: Path, fixed_root: Path) -> Path:
        for project_root in (buggy_root, fixed_root):
            resolved = project_root.resolve()
            for ancestor in (resolved, *resolved.parents):
                if ancestor.name == "bugs_workspace":
                    return ancestor

        try:
            return Path(os.path.commonpath([str(buggy_root.resolve()), str(fixed_root.resolve())]))
        except Exception:
            return buggy_root.resolve().parent

    def _container_path(self, host_path: Path) -> Path:
        resolved = Path(host_path).resolve()
        return CONTAINER_WORKSPACE_ROOT / resolved.relative_to(self.workspace_root)


# =============================================================
# Simple standalone test
# =============================================================

if __name__ == "__main__":
    adapter = Black2Adapter()

    if not SEED_INPUT.exists():
        print(f"Seed input not found: {SEED_INPUT}")
        sys.exit(1)

    source = SEED_INPUT.read_text(encoding="utf-8")
    result = adapter.evaluate(source)

    print()
    print("=== Black BugsInPy Bug #2 ===")
    print(f"Input     : {SEED_INPUT}")
    print(f"Size      : {result.candidate_size}")
    print(f"Outcome   : {result.outcome}")

    print()
    print("=== Buggy result ===")
    print(json.dumps(result.buggy_output, indent=2))

    print()
    print("=== Fixed result ===")
    print(json.dumps(result.fixed_output, indent=2))

    print()
    print(f"Buggy executed lines: {len(result.coverage)}")

    for filename, line in sorted(result.coverage)[:30]:
        print(f"  {filename}:{line}")
