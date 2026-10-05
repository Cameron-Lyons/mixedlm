from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm import glance, tidy

from tests._nlmm_models import fit_asymptotic_nlmm


@pytest.fixture(scope="module")
def nlmm_model():
    return fit_asymptotic_nlmm()


class TestTidyFixedEffects:
    def test_lmm_fixed_effect_table(self, sleepstudy_slopes_lmm) -> None:
        table = tidy(sleepstudy_slopes_lmm, ddf_method="normal")

        assert table["effect"].tolist() == ["fixed", "fixed"]
        assert table["term"].tolist() == ["(Intercept)", "Days"]
        assert np.allclose(table["estimate"], sleepstudy_slopes_lmm.beta)
        assert np.all(table["std.error"] > 0)
        assert np.all(table["p.value"].between(0, 1))
        assert table["df"].isna().all()

    def test_lmm_denominator_df_and_confidence_intervals(self, sleepstudy_slopes_lmm) -> None:
        table = sleepstudy_slopes_lmm.tidy(conf_int=True, ddf_method="Satterthwaite")

        assert np.all(np.isfinite(table["df"]))
        assert np.all(table["df"] > 0)
        assert np.all(table["conf.low"] < table["estimate"])
        assert np.all(table["conf.high"] > table["estimate"])

    def test_glmm_uses_wald_z_statistics(self, cbpp_glmm) -> None:
        table = tidy(cbpp_glmm, conf_int=True)

        assert set(table["term"]) == set(cbpp_glmm.matrices.fixed_names)
        assert table["df"].isna().all()
        assert np.all(table["p.value"].between(0, 1))
        assert np.all(table["conf.low"] <= table["estimate"])
        assert np.all(table["conf.high"] >= table["estimate"])

    def test_nlmm_uses_residual_df(self, nlmm_model) -> None:
        table = nlmm_model.tidy(conf_int=True)

        assert table["term"].tolist() == nlmm_model.model.param_names
        assert np.all(table["df"] == nlmm_model.df_residual())
        assert np.all(table["p.value"].dropna().between(0, 1))


class TestTidyRandomEffects:
    def test_random_parameters_include_sd_correlation_and_residual(
        self, sleepstudy_slopes_lmm
    ) -> None:
        table = tidy(sleepstudy_slopes_lmm, effects="ran_pars")

        assert set(table["effect"]) == {"ran_pars"}
        assert "sd__(Intercept)" in table["term"].tolist()
        assert "sd__Days" in table["term"].tolist()
        assert "cor__(Intercept).Days" in table["term"].tolist()
        residual = table.loc[table["group"] == "Residual", "estimate"]
        assert residual.iloc[0] == pytest.approx(sleepstudy_slopes_lmm.sigma)

    def test_random_values_include_levels_and_conditional_se(self, sleepstudy_slopes_lmm) -> None:
        table = tidy(sleepstudy_slopes_lmm, effects="ran_vals")

        assert len(table) == 18 * 2
        assert set(table["term"]) == {"(Intercept)", "Days"}
        assert table["level"].nunique() == 18
        assert np.all(table["std.error"] >= 0)

    def test_all_effects_have_stable_schema(self, sleepstudy_slopes_lmm) -> None:
        table = tidy(sleepstudy_slopes_lmm, effects="all", conf_int=True, ddf_method="normal")

        assert table.columns.tolist() == [
            "effect",
            "group",
            "level",
            "term",
            "estimate",
            "std.error",
            "statistic",
            "df",
            "p.value",
            "conf.low",
            "conf.high",
        ]
        assert set(table["effect"]) == {"fixed", "ran_pars", "ran_vals"}

    def test_nlmm_random_parameters_include_correlations(self, nlmm_model) -> None:
        table = nlmm_model.tidy(effects="ran_pars")

        terms = table["term"].tolist()
        assert "sd__Asym" in terms
        assert "sd__R0" in terms
        assert "cor__Asym.R0" in terms
        assert "sd__Observation" in terms
        assert np.all(np.isfinite(table["estimate"]))


class TestGlance:
    @pytest.mark.parametrize(
        ("fixture_name", "model_type", "family"),
        [
            ("sleepstudy_slopes_lmm", "lmer", "Gaussian"),
            ("cbpp_glmm", "glmer", "Binomial"),
            ("nlmm_model", "nlmer", "Gaussian"),
        ],
    )
    def test_common_model_summary(
        self,
        request: pytest.FixtureRequest,
        fixture_name: str,
        model_type: str,
        family: str,
    ) -> None:
        model = request.getfixturevalue(fixture_name)
        table = glance(model)

        assert len(table) == 1
        assert table.loc[0, "model"] == model_type
        assert table.loc[0, "family"] == family
        assert table.loc[0, "nobs"] == model.nobs()
        assert table.loc[0, "n_groups"] == sum(model.ngrps().values())
        assert table.loc[0, "AIC"] == pytest.approx(model.AIC())
        assert table.loc[0, "BIC"] == pytest.approx(model.BIC())
        assert bool(table.loc[0, "converged"]) is model.converged

    def test_method_matches_top_level_function(self, sleepstudy_slopes_lmm) -> None:
        pd.testing.assert_frame_equal(sleepstudy_slopes_lmm.glance(), glance(sleepstudy_slopes_lmm))


class TestReportingValidation:
    def test_rejects_unknown_effect(self, sleepstudy_slopes_lmm) -> None:
        with pytest.raises(ValueError, match="Unknown effect"):
            tidy(sleepstudy_slopes_lmm, effects="mystery")

    def test_rejects_invalid_confidence_level(self, sleepstudy_slopes_lmm) -> None:
        with pytest.raises(ValueError, match="between 0 and 1"):
            tidy(sleepstudy_slopes_lmm, conf_int=True, conf_level=1.0)

    def test_rejects_unknown_ddf_method(self, sleepstudy_slopes_lmm) -> None:
        with pytest.raises(ValueError, match="Unknown ddf_method"):
            tidy(sleepstudy_slopes_lmm, ddf_method="mystery")

    def test_rejects_non_model(self) -> None:
        with pytest.raises(TypeError, match="fitted mixed-model"):
            tidy(object())
        with pytest.raises(TypeError, match="fitted mixed-model"):
            glance(object())
