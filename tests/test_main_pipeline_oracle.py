import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_pipeline import build_differential_oracle


def test_build_differential_oracle_detects_buggy_vs_fixed_difference(tmp_path):
    buggy_path = tmp_path / "buggy_program.py"
    fixed_path = tmp_path / "fixed_program.py"

    buggy_path.write_text(
        "def run(candidate):\n    return sum(candidate) + 1 if any(v < 0 for v in candidate) else sum(candidate)\n",
        encoding="utf-8",
    )
    fixed_path.write_text("def run(candidate):\n    return sum(candidate)\n", encoding="utf-8")

    oracle = build_differential_oracle(str(buggy_path), str(fixed_path))

    assert oracle([1, 2, -3]) is True
    assert oracle([1, 2, 3]) is False
