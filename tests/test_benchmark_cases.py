import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_cases import _is_instrumentable_source


def test_is_instrumentable_source_skips_hidden_and_vendor_dirs(tmp_path):
    root = tmp_path
    assert _is_instrumentable_source(root / "pkg" / "module.py", root)
    assert not _is_instrumentable_source(root / "pkg" / "module_instrumented.py", root)
    assert not _is_instrumentable_source(root / ".eggs" / "pkg" / "module.py", root)
    assert not _is_instrumentable_source(root / "pkg" / "__pycache__" / "module.py", root)
    assert not _is_instrumentable_source(root / "pkg" / "tests" / "module.py", root)
    assert not _is_instrumentable_source(root / "pkg" / ".hidden" / "module.py", root)
