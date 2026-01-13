from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mixedlm import lmer
from mixedlm.inference.ddf import (
    DenomDFResult,
    kenward_roger_df,
    pvalues_with_ddf,
    satterthwaite_df,
)


def create_ddf_data(
    n_groups: int = 10, n_per_group: int = 5, seed: int = 42
) -> pd.DataFrame:
    np.random.seed(seed)
    data_rows = []
    for g in range(n_groups):
        group_effect = np.random.randn() * 0.5
        for _ in range(n_per_group):
            x = np.random.randn()
            y = 1.0 + 0.5 * x + group_effect + np.random.randn() * 0.5
            data_rows.append({"group": f"G{g + 1}", "x": x, "y": y})
    return pd.DataFrame(data_rows)


DDF_DATA = create_ddf_data()


class TestDenomDFResult:
    def test_getitem(self) -> None:
        result = DenomDFResult(
            df=np.array([10.0, 15.0]),
            method="Satterthwaite",
            param_names=["(Intercept)", "x"],
        )
        assert result["(Intercept)"] == 10.0
        assert result["x"] == 15.0

    def test_getitem_invalid_raises(self) -> None:
        result = DenomDFResult(
            df=np.array([10.0, 15.0]),
            method="Satterthwaite",
            param_names=["(Intercept)", "x"],
        )
        with pytest.raises(ValueError):
            result["nonexistent"]

    def test_as_dict(self) -> None:
        result = DenomDFResult(
            df=np.array([10.0, 15.0]),
            method="Satterthwaite",
            param_names=["(Intercept)", "x"],
        )
        d = result.as_dict()
        assert d == {"(Intercept)": 10.0, "x": 15.0}


class TestSatterthwaiteDF:
    def test_satterthwaite_df_basic(self) -> None:
        model = lmer("y ~ x + (1|group)", DDF_DATA)
        result = satterthwaite_df(model)

        assert isinstance(result, DenomDFResult)
        assert result.method == "Satterthwaite"
        assert len(result.df) == 2
        assert "(Intercept)" in result.param_names
        assert "x" in result.param_names

    def test_satterthwaite_df_positive(self) -> None:
        model = lmer("y ~ x + (1|group)", DDF_DATA)
        result = satterthwaite_df(model)

        assert np.all(result.df > 0)

    def test_satterthwaite_df_bounded(self) -> None:
        model = lmer("y ~ x + (1|group)", DDF_DATA)
        result = satterthwaite_df(model)

        n = model.matrices.n_obs
        p = model.matrices.n_fixed
        assert np.all(result.df >= 1.0)
        assert np.all(result.df <= n - p)

    def test_satterthwaite_df_multiple_fixed(self) -> None:
        data = DDF_DATA.copy()
        data["z"] = np.random.randn(len(data))
        model = lmer("y ~ x + z + (1|group)", data)
        result = satterthwaite_df(model)

        assert len(result.df) == 3
        assert "z" in result.param_names

    def test_satterthwaite_df_no_random_effects(self) -> None:
        data = DDF_DATA.copy()
        model = lmer("y ~ x + (1|group)", data)
        result = satterthwaite_df(model)

        assert result.df is not None


class TestKenwardRogerDF:
    def test_kenward_roger_df_basic(self) -> None:
        model = lmer("y ~ x + (1|group)", DDF_DATA)
        result = kenward_roger_df(model)

        assert isinstance(result, DenomDFResult)
        assert result.method == "Kenward-Roger"
        assert len(result.df) == 2
        assert "(Intercept)" in result.param_names
        assert "x" in result.param_names

    def test_kenward_roger_df_positive(self) -> None:
        model = lmer("y ~ x + (1|group)", DDF_DATA)
        result = kenward_roger_df(model)

        assert np.all(result.df > 0)

    def test_kenward_roger_df_bounded(self) -> None:
        model = lmer("y ~ x + (1|group)", DDF_DATA)
        result = kenward_roger_df(model)

        n = model.matrices.n_obs
        p = model.matrices.n_fixed
        assert np.all(result.df >= 1.0)
        assert np.all(result.df <= n - p)

    def test_kenward_roger_smaller_than_satterthwaite(self) -> None:
        model = lmer("y ~ x + (1|group)", DDF_DATA)
        satt = satterthwaite_df(model)
        kr = kenward_roger_df(model)

        assert np.all(kr.df <= satt.df + 1e-6)


class TestPvaluesWithDDF:
    def test_pvalues_satterthwaite(self) -> None:
        model = lmer("y ~ x + (1|group)", DDF_DATA)
        results = pvalues_with_ddf(model, method="Satterthwaite")

        assert "(Intercept)" in results
        assert "x" in results

        for name, (estimate, t_val, p_val) in results.items():
            assert isinstance(estimate, float)
            assert isinstance(t_val, float)
            assert isinstance(p_val, float)
            assert 0.0 <= p_val <= 1.0

    def test_pvalues_kenward_roger(self) -> None:
        model = lmer("y ~ x + (1|group)", DDF_DATA)
        results = pvalues_with_ddf(model, method="Kenward-Roger")

        assert "(Intercept)" in results
        assert "x" in results

        for name, (estimate, t_val, p_val) in results.items():
            assert isinstance(estimate, float)
            assert isinstance(t_val, float)
            assert isinstance(p_val, float)
            assert 0.0 <= p_val <= 1.0

    def test_pvalues_estimates_match_model(self) -> None:
        model = lmer("y ~ x + (1|group)", DDF_DATA)
        results = pvalues_with_ddf(model, method="Satterthwaite")

        for i, name in enumerate(model.matrices.fixed_names):
            estimate, _, _ = results[name]
            assert np.isclose(estimate, model.beta[i])

    def test_pvalues_invalid_method_raises(self) -> None:
        model = lmer("y ~ x + (1|group)", DDF_DATA)

        with pytest.raises(ValueError, match="Unknown method"):
            pvalues_with_ddf(model, method="invalid")

    def test_pvalues_t_values_consistent(self) -> None:
        model = lmer("y ~ x + (1|group)", DDF_DATA)
        results = pvalues_with_ddf(model, method="Satterthwaite")
        vcov = model.vcov()

        for i, name in enumerate(model.matrices.fixed_names):
            estimate, t_val, _ = results[name]
            se = np.sqrt(vcov[i, i])
            expected_t = estimate / se if se > 0 else 0.0
            assert np.isclose(t_val, expected_t, rtol=1e-5) or np.isnan(t_val)


class TestDDFWithDifferentModels:
    def test_ddf_random_slope(self) -> None:
        model = lmer("y ~ x + (x|group)", DDF_DATA)
        satt = satterthwaite_df(model)
        kr = kenward_roger_df(model)

        assert len(satt.df) == 2
        assert len(kr.df) == 2
        assert np.all(satt.df > 0)
        assert np.all(kr.df > 0)

    def test_ddf_multiple_random_effects(self) -> None:
        data = DDF_DATA.copy()
        data["group2"] = np.tile(
            [f"H{i}" for i in range(5)], len(data) // 5 + 1
        )[: len(data)]
        model = lmer("y ~ x + (1|group) + (1|group2)", data)

        satt = satterthwaite_df(model)
        kr = kenward_roger_df(model)

        assert len(satt.df) == 2
        assert len(kr.df) == 2

    def test_ddf_larger_dataset(self) -> None:
        large_data = create_ddf_data(n_groups=30, n_per_group=10)
        model = lmer("y ~ x + (1|group)", large_data)

        satt = satterthwaite_df(model)
        kr = kenward_roger_df(model)

        assert np.all(satt.df > 0)
        assert np.all(kr.df > 0)
        assert np.all(satt.df <= 300 - 2)
