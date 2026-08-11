from __future__ import annotations

import subprocess
import sys


def test_top_level_import_does_not_load_matplotlib() -> None:
    script = """
import builtins

real_import = builtins.__import__


def guarded_import(name, *args, **kwargs):
    if name == "matplotlib" or name.startswith("matplotlib."):
        raise AssertionError(f"unexpected eager import: {name}")
    return real_import(name, *args, **kwargs)


builtins.__import__ = guarded_import
import mixedlm
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
