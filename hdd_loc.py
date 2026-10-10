"""Compatibility wrapper for the HDD-LOC implementation.

The repository currently exposes the core implementation in hdd_algorithm.py,
but the Defects4J bridge and the line-localization modules expect an
``hdd_loc`` module name.
"""

from hdd_algorithm import HDDResult, HierarchicalDeltaDebugger, Node, TestRecord
from hdd_ddmin_loc import HDDDDMinLocDebugger
from parser_wdd import PythonParserWeightedDeltaDebugger

__all__ = [
    "HDDResult",
    "HierarchicalDeltaDebugger",
    "HDDDDMinLocDebugger",
    "PythonParserWeightedDeltaDebugger",
    "Node",
    "TestRecord",
]
