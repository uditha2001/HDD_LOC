from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bugsinpy_case_adapter import BugsInPyCaseAdapter


def _adapter(tmp_path):
    buggy = tmp_path / "buggy"
    fixed = tmp_path / "fixed"
    buggy.mkdir()
    fixed.mkdir()
    return BugsInPyCaseAdapter("tqdm_1", buggy, fixed, "tqdm", collect_coverage=False), buggy


def test_different_successful_outputs_are_fail(tmp_path):
    adapter, buggy_root = _adapter(tmp_path)
    adapter._run_version = lambda root, *args: {  # type: ignore[method-assign]
        "kind": "value", "output": "buggy" if root == buggy_root else "fixed"
    }

    result = adapter.evaluate({"input": [1, 2]})
    assert result.outcome == "FAIL"
    assert result.semantically_valid is True


def test_same_exception_is_invalid_not_pass(tmp_path):
    adapter, _ = _adapter(tmp_path)
    adapter._run_version = lambda *args: {  # type: ignore[method-assign]
        "kind": "exception", "exception_type": "KeyError", "message": "missing"
    }

    result = adapter.evaluate({})
    assert result.outcome == "INVALID"
    assert result.semantically_valid is False
