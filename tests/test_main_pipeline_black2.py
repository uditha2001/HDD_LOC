from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main_pipeline


def test_build_black2_oracle_wraps_adapter(monkeypatch):
    captured = {}

    class FakeAdapter:
        def __init__(self, buggy_root, fixed_root, timeout, collect_coverage):
            captured["init"] = (buggy_root, fixed_root, timeout, collect_coverage)

        def oracle(self, candidate_source):
            captured["candidate_source"] = candidate_source
            return candidate_source == "abc"

    monkeypatch.setattr(main_pipeline, "Black2Adapter", FakeAdapter)

    oracle, adapter = main_pipeline.build_black2_oracle(
        buggy_root="/buggy",
        fixed_root="/fixed",
        timeout=7,
        collect_coverage=False,
    )

    assert isinstance(adapter, FakeAdapter)
    assert captured["init"] == ("/buggy", "/fixed", 7, False)
    assert oracle(["a", "b", "c"]) is True
    assert captured["candidate_source"] == "abc"


def test_black2_source_is_split_into_preserving_lines():
    assert main_pipeline._black2_source_to_structured_input("a\nb\n") == ["a\n", "b\n"]
