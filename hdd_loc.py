"""Compatibility wrapper for the HDD-LOC implementation.

The repository currently exposes the core implementation in hdd_algorithm.py,
but the Defects4J bridge and the line-localization modules expect an
``hdd_loc`` module name.
"""

from hdd_algorithm import HDDResult, HierarchicalDeltaDebugger, Node, TestRecord

__all__ = ["HDDResult", "HierarchicalDeltaDebugger", "Node", "TestRecord"]
