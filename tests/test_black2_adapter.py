from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from black2_adapter import Black2Adapter


def test_evaluate_reports_fail_when_bug_is_preserved(tmp_path):
    buggy_root = tmp_path / "buggy"
    fixed_root = tmp_path / "fixed"
    buggy_root.mkdir()
    fixed_root.mkdir()

    adapter = Black2Adapter(buggy_root=buggy_root, fixed_root=fixed_root, collect_coverage=False)

    def fake_run_version(project_root, candidate_file, runner_file, coverage_file, use_coverage):
        if project_root == buggy_root:
            return {"kind": "formatted", "output": "buggy-output"}
        return {"kind": "formatted", "output": "fixed-output"}

    adapter._run_version = fake_run_version  # type: ignore[method-assign]

    result = adapter.evaluate("print('hello')\n")

    assert result.outcome == "FAIL"
    assert result.buggy_output["output"] == "buggy-output"
    assert result.fixed_output["output"] == "fixed-output"
    assert result.coverage == set()
    assert result.candidate_size == len("print('hello')\n")


def test_evaluate_reports_pass_when_outputs_match(tmp_path):
    buggy_root = tmp_path / "buggy"
    fixed_root = tmp_path / "fixed"
    buggy_root.mkdir()
    fixed_root.mkdir()

    adapter = Black2Adapter(buggy_root=buggy_root, fixed_root=fixed_root, collect_coverage=False)

    def fake_run_version(project_root, candidate_file, runner_file, coverage_file, use_coverage):
        return {"kind": "formatted", "output": "same-output"}

    adapter._run_version = fake_run_version  # type: ignore[method-assign]

    result = adapter.evaluate("print('hello')\n")

    assert result.outcome == "PASS"
    assert result.buggy_output == result.fixed_output


def test_run_version_builds_docker_exec_command(tmp_path, monkeypatch):
    workspace_root = tmp_path / "bugs_workspace"
    buggy_root = workspace_root / "black_2_buggy" / "black"
    fixed_root = workspace_root / "black_2_fixed" / "black"
    buggy_root.mkdir(parents=True)
    fixed_root.mkdir(parents=True)

    temp_dir = workspace_root / "tmp"
    temp_dir.mkdir(parents=True)

    candidate_file = temp_dir / "candidate.py"
    runner_file = temp_dir / "runner.py"
    candidate_file.write_text("print('x')\n", encoding="utf-8")
    runner_file.write_text("print('runner')\n", encoding="utf-8")

    adapter = Black2Adapter(buggy_root=buggy_root, fixed_root=fixed_root, collect_coverage=False)

    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout='{"kind": "formatted", "output": "ok"}', stderr="")

    monkeypatch.setattr("black2_adapter.subprocess.run", fake_run)

    adapter._run_version(
        project_root=buggy_root,
        candidate_file=candidate_file,
        runner_file=runner_file,
        coverage_file=None,
        use_coverage=False,
    )

    command = captured["command"]
    assert command[:5] == ["docker", "exec", "bugsinpy_bg", "bash", "-c"]
    assert "export PATH=$PATH:/BugsInPy/framework/bin:/framework/bin;" in command[5]
    assert "cd /workspace/tmp &&" in command[5]
    assert "/workspace/black_2_buggy/black" in command[5]
    assert "/workspace/tmp/candidate.py" in command[5]
