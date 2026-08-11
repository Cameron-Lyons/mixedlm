from __future__ import annotations

import subprocess
import sys
import textwrap


def _run_isolated(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_core_import_does_not_load_plotting_stack() -> None:
    result = _run_isolated(
        """
        import sys
        import mixedlm

        assert "matplotlib" not in sys.modules
        assert "nlopt" not in sys.modules
        assert callable(mixedlm.diagnostics.plot_qq)
        """
    )

    assert result.returncode == 0, result.stderr


def test_core_import_works_without_matplotlib() -> None:
    result = _run_isolated(
        """
        import builtins

        real_import = builtins.__import__

        def without_optional_dependencies(name, *args, **kwargs):
            if name in {"matplotlib", "nlopt"} or name.startswith("matplotlib."):
                raise ImportError("blocked optional dependency")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = without_optional_dependencies

        import mixedlm
        from mixedlm.estimation.optimizers import available_optimizers, has_nlopt

        assert not has_nlopt()
        assert all(not name.startswith("nloptwrap_") for name in available_optimizers())

        try:
            mixedlm.diagnostics.plot_qq(None)
        except ImportError as exc:
            assert "mixedlm[plots]" in str(exc)
        else:
            raise AssertionError("plotting should require matplotlib")
        """
    )

    assert result.returncode == 0, result.stderr
