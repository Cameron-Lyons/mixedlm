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


def test_root_import_does_not_load_modeling_stack() -> None:
    result = _run_isolated(
        """
        import sys
        import mixedlm

        package_modules = {name for name in sys.modules if name.startswith("mixedlm")}
        assert package_modules == {"mixedlm"}
        assert not {"numpy", "pandas", "scipy", "matplotlib"} & sys.modules.keys()
        assert len(mixedlm.__all__) == len(set(mixedlm.__all__))
        assert set(mixedlm.__all__) == set(mixedlm._LAZY_EXPORTS)
        assert set(mixedlm.__all__) <= set(dir(mixedlm))
        """
    )

    assert result.returncode == 0, result.stderr


def test_lightweight_formula_export_stays_independent() -> None:
    result = _run_isolated(
        """
        import sys
        import mixedlm

        parser = mixedlm.parse_formula
        formula = parser("y ~ x + (1 | group)")

        assert str(formula) == "y ~ x + (1 | group)"
        assert parser is mixedlm.parse_formula
        assert "mixedlm.formula.parser" in sys.modules
        assert "mixedlm.models.lmer" not in sys.modules
        assert not {"numpy", "pandas", "scipy", "matplotlib"} & sys.modules.keys()
        """
    )

    assert result.returncode == 0, result.stderr


def test_fitting_export_matches_direct_symbol() -> None:
    result = _run_isolated(
        """
        import mixedlm
        from mixedlm.models.lmer import lmer as direct_lmer

        assert mixedlm.lmer is direct_lmer
        assert mixedlm.lmer is mixedlm.lmer
        assert "lmer" in mixedlm.__dict__
        """
    )

    assert result.returncode == 0, result.stderr


def test_module_exports_are_cached() -> None:
    result = _run_isolated(
        """
        import mixedlm

        families = mixedlm.families
        assert families.__name__ == "mixedlm.families"
        assert families is mixedlm.families
        assert "families" in mixedlm.__dict__
        assert families.Binomial is mixedlm.families.Binomial
        """
    )

    assert result.returncode == 0, result.stderr


def test_unknown_root_attribute_raises_attribute_error() -> None:
    result = _run_isolated(
        """
        import mixedlm

        try:
            mixedlm.not_a_public_name
        except AttributeError as exc:
            assert "not_a_public_name" in str(exc)
        else:
            raise AssertionError("unknown attributes must fail")
        """
    )

    assert result.returncode == 0, result.stderr
