from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution_records import SourceLocation
from ground_truth import deleted_buggy_lines, executable_ground_truth


def test_extracts_only_buggy_side_changed_lines(tmp_path):
    patch = tmp_path / "bug.patch"
    patch.write_text(
        "diff --git a/pkg/code.py b/pkg/code.py\n"
        "--- a/pkg/code.py\n"
        "+++ b/pkg/code.py\n"
        "@@ -10,4 +10,4 @@\n"
        " context\n"
        "-old one\n"
        "-old two\n"
        "+new\n"
        " context\n",
        encoding="utf-8",
    )

    assert deleted_buggy_lines(patch) == frozenset(
        {SourceLocation("pkg/code.py", 11), SourceLocation("pkg/code.py", 12)}
    )


def test_maps_multiline_patch_to_executable_statement_start(tmp_path):
    source_root = tmp_path / "buggy"
    source_root.mkdir()
    (source_root / "code.py").write_text(
        "result = (\n    first\n    and second\n)\n", encoding="utf-8"
    )

    mapped = executable_ground_truth(
        frozenset({SourceLocation("code.py", 3)}), source_root
    )

    assert mapped == frozenset({SourceLocation("code.py", 1)})
