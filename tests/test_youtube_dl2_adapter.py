from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from youtube_dl2_adapter import YoutubeDL2Adapter


def _adapter(tmp_path):
    buggy = tmp_path / "buggy"
    fixed = tmp_path / "fixed"
    buggy.mkdir()
    fixed.mkdir()
    return YoutubeDL2Adapter(buggy, fixed, collect_coverage=False), buggy, fixed


def test_differing_format_lists_are_semantic_failures(tmp_path):
    adapter, buggy_root, _ = _adapter(tmp_path)

    def fake_run(project_root, candidate_file, runner_file, coverage_file, use_coverage):
        count = 6 if project_root == buggy_root else 7
        return {"kind": "formats", "output": [{"id": number} for number in range(count)]}

    adapter._run_version = fake_run  # type: ignore[method-assign]
    result = adapter.evaluate("<MPD/>")

    assert result.outcome == "FAIL"
    assert result.structurally_valid is True
    assert result.semantically_valid is True


def test_matching_format_lists_pass(tmp_path):
    adapter, _, _ = _adapter(tmp_path)
    adapter._run_version = lambda *args: {"kind": "formats", "output": []}  # type: ignore[method-assign]

    assert adapter.evaluate("<MPD/>").outcome == "PASS"


def test_both_revisions_rejecting_candidate_is_invalid(tmp_path):
    adapter, _, _ = _adapter(tmp_path)
    adapter._run_version = lambda *args: {  # type: ignore[method-assign]
        "kind": "exception",
        "exception_type": "ValueError",
        "message": "not a usable MPD",
    }

    result = adapter.evaluate("<MPD/>")
    assert result.outcome == "INVALID"
    assert result.structurally_valid is True
    assert result.semantically_valid is False
