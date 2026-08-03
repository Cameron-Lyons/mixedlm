from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm import families, glance, glmer, lmer, nlme, nlmer, tidy

from tests._lmer_data import CBPP, SLEEPSTUDY


@pytest.fixture(scope="module")
def lmm_model():
    return lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY)


@pytest.fixture(scope="module")
def glmm_model():
    return glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial())


@pytest.fixture(scope="module")
def nlmm_model():
    rng = np.random.default_rng(814)
    rows = []
    for group_index in range(6):
        asym = 200.0 + rng.normal(0.0, 12.0)
        r0 = 175.0 + rng.normal(0.0, 6.0)
        lrc = -2.8 + rng.normal(0.0, 0.1)
        for time in np.linspace(0.0, 10.0, 10):
            response = asym - (asym - r0) * np.exp(-np.exp(lrc) * time)
            rows.append(
                {
                    "subject": f"S{group_index + 1}",
                    "time": time,
                    "response": response + rng.normal(0.0, 3.0),
                }
            )
    data = pd.DataFrame(rows)
    return nlmer(
        nlme.SSasymp(),
        data,
        x_var="time",
        y_var="response",
        group_var="subject",
    )


class TestTidyFixedEffects:
    def test_lmm_fixed_effect_table(self, lmm_model) -> None:
        table = tidy(lmm_model, ddf_method="normal")

        assert table["effect"].tolist() == ["fixed", "fixed"]
        assert table["term"].tolist() == ["(Intercept)", "Days"]
        assert np.allclose(table["estimate"], lmm_model.beta)
        assert np.all(table["std.error"] > 0)
        assert np.all(table["p.value"].between(0, 1))
        assert table["df"].isna().all()

    def test_lmm_denominator_df_and_confidence_intervals(self, lmm_model) -> None:
        table = lmm_model.tidy(conf_int=True, ddf_method="Satterthwaite")

        assert np.all(np.isfinite(table["df"]))
        assert np.all(table["df"] > 0)
        assert np.all(table["conf.low"] < table["estimate"])
        assert np.all(table["conf.high"] > table["estimate"])

    def test_glmm_uses_wald_z_statistics(self, glmm_model) -> None:
        table = tidy(glmm_model, conf_int=True)

        assert set(table["term"]) == set(glmm_model.matrices.fixed_names)
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
    def test_random_parameters_include_sd_correlation_and_residual(self, lmm_model) -> None:
        table = tidy(lmm_model, effects="ran_pars")

        assert set(table["effect"]) == {"ran_pars"}
        assert "sd__(Intercept)" in table["term"].tolist()
        assert "sd__Days" in table["term"].tolist()
        assert "cor__(Intercept).Days" in table["term"].tolist()
        residual = table.loc[table["group"] == "Residual", "estimate"]
        assert residual.iloc[0] == pytest.approx(lmm_model.sigma)

    def test_random_values_include_levels_and_conditional_se(self, lmm_model) -> None:
        table = tidy(lmm_model, effects="ran_vals")

        assert len(table) == 18 * 2
        assert set(table["term"]) == {"(Intercept)", "Days"}
        assert table["level"].nunique() == 18
        assert np.all(table["std.error"] >= 0)

    def test_all_effects_have_stable_schema(self, lmm_model) -> None:
        table = tidy(lmm_model, effects="all", conf_int=True, ddf_method="normal")

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
            ("lmm_model", "lmer", "Gaussian"),
            ("glmm_model", "glmer", "Binomial"),
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

    def test_method_matches_top_level_function(self, lmm_model) -> None:
        pd.testing.assert_frame_equal(lmm_model.glance(), glance(lmm_model))


class TestReportingValidation:
    def test_rejects_unknown_effect(self, lmm_model) -> None:
        with pytest.raises(ValueError, match="Unknown effect"):
            tidy(lmm_model, effects="mystery")

    def test_rejects_invalid_confidence_level(self, lmm_model) -> None:
        with pytest.raises(ValueError, match="between 0 and 1"):
            tidy(lmm_model, conf_int=True, conf_level=1.0)

    def test_rejects_unknown_ddf_method(self, lmm_model) -> None:
        with pytest.raises(ValueError, match="Unknown ddf_method"):
            tidy(lmm_model, ddf_method="mystery")

    def test_rejects_non_model(self) -> None:
        with pytest.raises(TypeError, match="fitted mixed-model"):
            tidy(object())
        with pytest.raises(TypeError, match="fitted mixed-model"):
            glance(object())
