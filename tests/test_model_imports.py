from __future__ import annotations

import subprocess
import sys
import textwrap


def test_model_entry_points_defer_scipy_stats() -> None:
    code = """
        import sys
        import mixedlm

        assert callable(mixedlm.lmer) and callable(mixedlm.glmer) and callable(mixedlm.nlmer)
        assert "mixedlm.models.lmer" in sys.modules
        assert "scipy.stats" not in sys.modules
        """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
