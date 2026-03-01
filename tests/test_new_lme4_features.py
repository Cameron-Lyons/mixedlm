"""Tests for newly implemented lme4 features."""

import numpy as np
import pandas as pd
import pytest
from mixedlm import lmer


def create_sleepstudy():
    """Create sleepstudy-like data."""
    np.random.seed(42)
    subjects = [f"S{i:02d}" for i in range(18)]
    days = list(range(10))

    data = []
    for subj in subjects:
        intercept = 250 + np.random.normal(0, 25)
        slope = 10 + np.random.normal(0, 5)
        for day in days:
            reaction = intercept + slope * day + np.random.normal(0, 25)
            data.append({"Subject": subj, "Days": day, "Reaction": reaction})

    return pd.DataFrame(data)


SLEEPSTUDY = create_sleepstudy()


class TestSummaryPValues:
    """Tests for enhanced summary() with p-values."""

    def test_summary_with_satterthwaite(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        summary = result.summary(ddf_method="Satterthwaite")
        assert "Pr(>|t|)" in summary
        assert "df" in summary.lower()
        assert "***" in summary or "**" in summary or "*" in summary

    def test_summary_with_kenward_roger(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        summary = result.summary(ddf_method="Kenward-Roger")
        assert "Pr(>|t|)" in summary

    def test_summary_without_pvalues(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        summary = result.summary(ddf_method=None)
        assert "Pr(>|t|)" not in summary

    def test_summary_invalid_method(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        with pytest.raises(ValueError, match="Unknown ddf_method"):
            result.summary(ddf_method="invalid")


class TestGetMEExtended:
    """Tests for extended getME() functionality."""

    def test_getme_rx(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        RX = result.getME("RX")
        p = result.matrices.n_fixed
        assert RX.shape == (p, p)
        assert np.allclose(RX, np.triu(RX))

    def test_getme_rzx(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        RZX = result.getME("RZX")
        q = result.matrices.n_random
        p = result.matrices.n_fixed
        assert RZX.shape[0] == q
        assert RZX.shape[1] == p

    def test_getme_lind(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        Lind = result.getME("Lind")
        assert isinstance(Lind, np.ndarray)
        assert Lind.dtype == np.int64

    def test_getme_devcomp(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        devcomp = result.getME("devcomp")
        assert "cmp" in devcomp
        assert "dims" in devcomp
        assert "ldL2" in devcomp["cmp"]
        assert "wrss" in devcomp["cmp"]
        assert "pwrss" in devcomp["cmp"]
        assert "n" in devcomp["dims"]
        assert "p" in devcomp["dims"]
        assert "q" in devcomp["dims"]


class TestAnovaType3:
    """Tests for Type III ANOVA."""

    def test_anova_type3_basic(self):
        from mixedlm.inference.anova import anova_type3

        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        anova = anova_type3(result)
        assert "Days" in anova.terms
        assert len(anova.terms) >= 1
        assert all(anova.p_value <= 1.0)
        assert all(anova.f_value >= 0)

    def test_anova_type3_with_kr(self):
        from mixedlm.inference.anova import anova_type3

        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        anova = anova_type3(result, ddf_method="Kenward-Roger")
        assert anova.ddf_method == "Kenward-Roger"

    def test_anova_type3_str(self):
        from mixedlm.inference.anova import anova_type3

        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        anova = anova_type3(result)
        s = str(anova)
        assert "Type III" in s
        assert "F value" in s


class TestAdaptiveGH:
    """Tests for adaptive Gauss-Hermite quadrature."""

    def test_agh_nAGQ_1_equals_laplace(self):
        from mixedlm.estimation.laplace import adaptive_gh_deviance, laplace_deviance
        from mixedlm.families import Binomial
        from mixedlm.formula.parser import parse_formula
        from mixedlm.matrices.design import build_model_matrices

        np.random.seed(42)
        n = 100
        group = np.repeat(np.arange(10), 10)
        x = np.random.randn(n)
        prob = 1 / (1 + np.exp(-(x + 0.5)))
        y = np.random.binomial(1, prob)
        data = pd.DataFrame({"y": y, "x": x, "group": [f"g{g}" for g in group]})

        formula = parse_formula("y ~ x + (1 | group)")
        matrices = build_model_matrices(formula, data)
        family = Binomial()
        theta = np.array([1.0])

        dev_laplace, _, _ = laplace_deviance(theta, matrices, family)
        dev_agh1, _, _ = adaptive_gh_deviance(theta, matrices, family, nAGQ=1)

        assert np.isclose(dev_laplace, dev_agh1, rtol=1e-6)

    def test_agh_nAGQ_greater_than_1(self):
        from mixedlm.estimation.laplace import adaptive_gh_deviance
        from mixedlm.families import Binomial
        from mixedlm.formula.parser import parse_formula
        from mixedlm.matrices.design import build_model_matrices

        np.random.seed(42)
        n = 100
        group = np.repeat(np.arange(10), 10)
        x = np.random.randn(n)
        prob = 1 / (1 + np.exp(-(x + 0.5)))
        y = np.random.binomial(1, prob)
        data = pd.DataFrame({"y": y, "x": x, "group": [f"g{g}" for g in group]})

        formula = parse_formula("y ~ x + (1 | group)")
        matrices = build_model_matrices(formula, data)
        family = Binomial()
        theta = np.array([1.0])

        dev_agh5, beta, u = adaptive_gh_deviance(theta, matrices, family, nAGQ=5)
        assert np.isfinite(dev_agh5)
        assert len(beta) == matrices.n_fixed
        assert len(u) == matrices.n_random

    def test_agh_parallel_matches_serial(self):
        from mixedlm.estimation.laplace import adaptive_gh_deviance
        from mixedlm.families import Binomial
        from mixedlm.formula.parser import parse_formula
        from mixedlm.matrices.design import build_model_matrices

        np.random.seed(42)
        n = 100
        group = np.repeat(np.arange(10), 10)
        x = np.random.randn(n)
        prob = 1 / (1 + np.exp(-(x + 0.5)))
        y = np.random.binomial(1, prob)
        data = pd.DataFrame({"y": y, "x": x, "group": [f"g{g}" for g in group]})

        formula = parse_formula("y ~ x + (1 | group)")
        matrices = build_model_matrices(formula, data)
        family = Binomial()
        theta = np.array([1.0])

        dev_serial, beta_serial, u_serial = adaptive_gh_deviance(
            theta, matrices, family, nAGQ=5, n_jobs=1
        )
        dev_parallel, beta_parallel, u_parallel = adaptive_gh_deviance(
            theta, matrices, family, nAGQ=5, n_jobs=2
        )

        assert np.isclose(dev_serial, dev_parallel, rtol=1e-8, atol=1e-8)
        assert np.allclose(beta_serial, beta_parallel, rtol=1e-10, atol=1e-10)
        assert np.allclose(u_serial, u_parallel, rtol=1e-10, atol=1e-10)


class TestDevianceComponents:
    """Tests for deviance components."""

    def test_get_deviance_components(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        dc = result.get_deviance_components()
        assert hasattr(dc, "total")
        assert hasattr(dc, "ldL2")
        assert hasattr(dc, "ldRX2")
        assert hasattr(dc, "wrss")
        assert hasattr(dc, "ussq")
        assert hasattr(dc, "pwrss")
        assert hasattr(dc, "sigma2")

    def test_deviance_components_sum(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        dc = result.get_deviance_components()
        assert np.isclose(dc.total, result.deviance, rtol=1e-4)

    def test_deviance_components_str(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        dc = result.get_deviance_components()
        s = str(dc)
        assert "Deviance Components" in s
        assert "wrss" in s.lower()


class TestCustomFamilies:
    """Tests for custom family helpers."""

    def test_validate_binomial(self):
        from mixedlm.families import Binomial
        from mixedlm.families.custom import validate_family

        assert validate_family(Binomial())

    def test_validate_poisson(self):
        from mixedlm.families import Poisson
        from mixedlm.families.custom import validate_family

        assert validate_family(Poisson())

    def test_quasi_family(self):
        from mixedlm.families import Poisson
        from mixedlm.families.custom import QuasiFamily, validate_family

        quasi_poisson = QuasiFamily(Poisson(), phi=2.0)
        assert validate_family(quasi_poisson)

        mu = np.array([1.0, 2.0, 5.0])
        base_var = Poisson().variance(mu)
        quasi_var = quasi_poisson.variance(mu)
        assert np.allclose(quasi_var, 2.0 * base_var)

    def test_custom_family_base(self):
        from mixedlm.families.base import LogLink
        from mixedlm.families.custom import CustomFamily

        class MyFamily(CustomFamily):
            def __init__(self):
                self.link = LogLink()

            def variance(self, mu):
                return mu**1.5

            def deviance_resids(self, y, mu, wt):
                return 2 * wt * (y - mu)

        fam = MyFamily()
        assert hasattr(fam, "link")
        mu = np.array([1.0, 2.0, 3.0])
        assert fam.variance(mu).shape == mu.shape


class TestProfile2D:
    """Tests for 2D profile likelihood slices."""

    def test_slice2d_basic(self):
        from mixedlm.inference.profile import slice2D

        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        profile2d = slice2D(result, "(Intercept)", "Days", n_points=5)

        assert profile2d.param1 == "(Intercept)"
        assert profile2d.param2 == "Days"
        assert profile2d.zeta.shape == (5, 5)
        assert np.isfinite(profile2d.zeta).all()

    def test_slice2d_mle_at_center(self):
        from mixedlm.inference.profile import slice2D

        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        profile2d = slice2D(result, "(Intercept)", "Days", n_points=5)

        idx1 = result.matrices.fixed_names.index("(Intercept)")
        idx2 = result.matrices.fixed_names.index("Days")

        assert np.isclose(profile2d.mle1, result.beta[idx1])
        assert np.isclose(profile2d.mle2, result.beta[idx2])

    def test_slice2d_invalid_param(self):
        from mixedlm.inference.profile import slice2D

        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        with pytest.raises(ValueError, match="not found"):
            slice2D(result, "invalid_param", "Days")

    def test_slice2d_parallel_matches_serial(self):
        from mixedlm.inference.profile import slice2D

        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        profile_serial = slice2D(result, "(Intercept)", "Days", n_points=5, n_jobs=1)
        profile_parallel = slice2D(result, "(Intercept)", "Days", n_points=5, n_jobs=2)

        assert np.allclose(profile_serial.zeta, profile_parallel.zeta, rtol=1e-8, atol=1e-8)

    def test_slice2d_small_grid_skips_parallel_pool(self, monkeypatch):
        import mixedlm.inference.profile as profile_mod
        from mixedlm.inference.profile import slice2D

        class _FailExecutor:
            def __init__(self, *args, **kwargs):
                raise AssertionError("Process pool should not be created for small grids")

        monkeypatch.setattr(profile_mod, "ProcessPoolExecutor", _FailExecutor)

        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        profile2d = slice2D(result, "(Intercept)", "Days", n_points=5, n_jobs=2)
        assert profile2d.zeta.shape == (5, 5)


class TestExportsAndImports:
    """Tests that new exports are accessible."""

    def test_import_anova_type3(self):
        from mixedlm.inference.anova import AnovaType3Result, anova_type3

        assert callable(anova_type3)
        assert AnovaType3Result is not None

    def test_import_deviance_components(self):
        from mixedlm.estimation.reml import DevianceComponents, profiled_deviance_components

        assert callable(profiled_deviance_components)
        assert DevianceComponents is not None

    def test_import_custom_families(self):
        from mixedlm.families import CustomFamily, QuasiFamily, validate_family

        assert CustomFamily is not None
        assert QuasiFamily is not None
        assert callable(validate_family)

    def test_import_slice2d(self):
        from mixedlm.inference.profile import Profile2DResult, slice2D

        assert callable(slice2D)
        assert Profile2DResult is not None

    def test_import_adaptive_gh(self):
        from mixedlm.estimation.laplace import adaptive_gh_deviance

        assert callable(adaptive_gh_deviance)
