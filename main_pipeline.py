"""Main integration script for HDD-LOC.

The pipeline is:
1. HDD minimizes the structured input and records pass/fail test cases.
2. line_localization replays each test case and records executed lines.
3. sbfl_score ranks the most suspicious lines.
"""

from __future__ import annotations

import os
from importlib.util import module_from_spec, spec_from_file_location
from typing import Any, Optional, Tuple

from hdd_algorithm import HDDResult, HierarchicalDeltaDebugger
from line_localization import collect_line_coverage, test_cases_from_hdd_result
from sbfl_score import rank_lines, render_report


def _load_program(program_path: str):
    abs_path = os.path.abspath(program_path)
    spec = spec_from_file_location("_pipeline_buggy_program", abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load program at {program_path!r}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_oracle(program_path: str, reference_path: Optional[str] = None):
    """Build a Python differential oracle for the buggy program."""

    program_module = _load_program(program_path)
    reference_module = _load_program(reference_path or program_path.replace("_buggy", "_fixed"))

    def oracle(candidate: Any) -> bool:
        try:
            buggy_output = getattr(program_module, "run")(candidate)
            fixed_output = getattr(reference_module, "run")(candidate)
            return buggy_output != fixed_output
        except Exception:
            try:
                getattr(reference_module, "run")(candidate)
            except Exception:
                return True
            return True

    return oracle


def run_pipeline(
    structured_input: Any,
    program_path: str,
    reference_path: Optional[str] = None,
    entry_function: str = "run",
    formula: str = "ochiai",
    normalize: bool = True,
    top_n: Optional[int] = None,
) -> Tuple[HDDResult, list, str]:
    """Run the full HDD-LOC pipeline for the Python buggy/fixed comparison demo."""

    oracle = _build_oracle(program_path, reference_path=reference_path)
    debugger = HierarchicalDeltaDebugger(oracle=oracle, weighting="subtree_size")
    hdd_result = debugger.reduce(structured_input)
    test_cases = test_cases_from_hdd_result(hdd_result, debugger)
    coverage = collect_line_coverage(program_path, test_cases, entry_function=entry_function)
    ranking = rank_lines(coverage, formula=formula, normalize=normalize)
    report = render_report(program_path, ranking, top_n=top_n)
    return hdd_result, ranking, report


if __name__ == "__main__":
    sample_input = [3, -1, 5, -2, 8]
    root = os.path.dirname(os.path.abspath(__file__))
    program_path = os.path.join(root, "sample_buggy_program.py")
    reference_path = os.path.join(root, "sample_fixed_program.py")
    _, _, report = run_pipeline(sample_input, program_path, reference_path=reference_path, top_n=5)
    print(report)
