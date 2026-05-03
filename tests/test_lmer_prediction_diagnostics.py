# ruff: noqa: F401
import numpy as np
import pandas as pd
import pytest
from mixedlm import (
    anova,
    families,
    findbars,
    glmer,
    glmerControl,
    is_mixed_formula,
    lmer,
    lmerControl,
    nlme,
    nlmer,
    nobars,
    parse_formula,
    subbars,
)
from mixedlm.formula.terms import InteractionTerm, VariableTerm
from mixedlm.matrices import build_model_matrices
from mixedlm.models.control import GlmerControl, LmerControl

from tests._lmer_data import CBPP, SLEEPSTUDY


class TestPredict:
    def test_lmer_predict_no_newdata(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        pred = result.predict()
        fitted = result.fitted()

        assert np.allclose(pred, fitted)

    def test_lmer_predict_same_data(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        pred = result.predict(newdata=SLEEPSTUDY)
        fitted = result.fitted()

        assert np.allclose(pred, fitted)

    def test_lmer_predict_fixed_only(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        pred_fixed = result.predict(newdata=SLEEPSTUDY, re_form="NA")

        fixef = result.fixef()
        expected_fixed = fixef["(Intercept)"] + fixef["Days"] * SLEEPSTUDY["Days"].values
        assert np.allclose(pred_fixed, expected_fixed)

    def test_lmer_predict_new_levels_error(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        new_data = pd.DataFrame({"Reaction": [300.0], "Days": [5.0], "Subject": ["999"]})

        with pytest.raises(ValueError, match="New level"):
            result.predict(newdata=new_data)

    def test_lmer_predict_new_levels_allowed(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        new_data = pd.DataFrame({"Reaction": [300.0], "Days": [5.0], "Subject": ["999"]})

        pred = result.predict(newdata=new_data, allow_new_levels=True)
        fixef = result.fixef()
        expected = fixef["(Intercept)"] + fixef["Days"] * 5.0

        assert np.isclose(pred[0], expected)

    def test_lmer_predict_random_slope(self):
        ctrl = LmerControl(optimizer="L-BFGS-B")
        result = lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY, control=ctrl)

        subject = SLEEPSTUDY["Subject"].iloc[0]
        new_data = pd.DataFrame({"Reaction": [300.0], "Days": [5.0], "Subject": [subject]})

        pred = result.predict(newdata=new_data)
        pred_fixed = result.predict(newdata=new_data, re_form="NA")

        assert pred[0] != pred_fixed[0]

    def test_lmer_predict_new_levels_se_fit_random_slope(self):
        ctrl = LmerControl(optimizer="L-BFGS-B")
        result = lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY, control=ctrl)

        new_data = pd.DataFrame(
            {
                "Reaction": [300.0, 320.0],
                "Days": [2.0, 7.0],
                "Subject": ["new_subject_a", "new_subject_b"],
            }
        )

        pred = result.predict(newdata=new_data, allow_new_levels=True, se_fit=True)
        pred_fixed = result.predict(newdata=new_data, re_form="NA")

        assert pred.se_fit is not None
        assert np.all(np.isfinite(pred.se_fit))
        assert np.all(pred.se_fit > 0)
        assert np.allclose(pred.fit, pred_fixed)

    def test_glmer_predict_no_newdata(self):
        result = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial())
        pred = result.predict()
        fitted = result.fitted()

        assert np.allclose(pred, fitted)

    def test_glmer_predict_same_data(self):
        result = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial())
        pred = result.predict(newdata=CBPP)
        fitted = result.fitted()

        assert np.allclose(pred, fitted)

    def test_glmer_predict_fixed_only(self):
        result = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial())
        pred_fixed = result.predict(newdata=CBPP, re_form="NA")
        pred_full = result.predict(newdata=CBPP)

        assert not np.allclose(pred_fixed, pred_full)

    def test_glmer_predict_link_scale(self):
        result = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial())
        pred_response = result.predict(newdata=CBPP, type="response")
        pred_link = result.predict(newdata=CBPP, type="link")

        assert np.all(pred_response >= 0) and np.all(pred_response <= 1)
        assert not np.all(pred_link >= 0) or not np.all(pred_link <= 1)

    def test_glmer_predict_new_levels_allowed(self):
        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_effects = np.random.randn(n_groups) * 0.3
        eta = -0.5 + 0.5 * x + group_effects[group]
        p = 1 / (1 + np.exp(-eta))
        y = np.random.binomial(1, p)

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        result = glmer("y ~ x + (1 | group)", data, family=families.Binomial())

        new_data = pd.DataFrame({"y": [0], "x": [0.5], "group": ["999"]})
        pred = result.predict(newdata=new_data, allow_new_levels=True)

        assert len(pred) == 1
        assert 0 <= pred[0] <= 1


class TestNAAction:
    def test_lmer_na_omit(self) -> None:
        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_effects = np.random.randn(n_groups) * 0.5
        y = 2.0 + 1.5 * x + group_effects[group] + np.random.randn(n) * 0.5

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        data.loc[0, "y"] = np.nan
        data.loc[5, "x"] = np.nan
        data.loc[10, "group"] = np.nan

        result = lmer("y ~ x + (1 | group)", data, na_action="omit")

        assert result.matrices.n_obs == n - 3
        assert len(result.fitted()) == n - 3
        assert len(result.residuals()) == n - 3
        assert result.converged

    def test_lmer_na_exclude(self) -> None:
        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_effects = np.random.randn(n_groups) * 0.5
        y = 2.0 + 1.5 * x + group_effects[group] + np.random.randn(n) * 0.5

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        data.loc[0, "y"] = np.nan
        data.loc[5, "x"] = np.nan

        result = lmer("y ~ x + (1 | group)", data, na_action="exclude")

        assert result.matrices.n_obs == n - 2

        fitted_vals = result.fitted()
        assert len(fitted_vals) == n
        assert np.isnan(fitted_vals[0])
        assert np.isnan(fitted_vals[5])
        assert not np.isnan(fitted_vals[1])

        resid = result.residuals()
        assert len(resid) == n
        assert np.isnan(resid[0])
        assert np.isnan(resid[5])

    def test_lmer_na_fail(self) -> None:
        np.random.seed(42)
        n = 50
        group = np.repeat(np.arange(5), 10)
        x = np.random.randn(n)
        y = 2.0 + 1.5 * x + np.random.randn(n) * 0.5

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        data.loc[0, "y"] = np.nan

        with pytest.raises(ValueError, match="Missing values"):
            lmer("y ~ x + (1 | group)", data, na_action="fail")

    def test_lmer_no_na(self) -> None:
        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_effects = np.random.randn(n_groups) * 0.5
        y = 2.0 + 1.5 * x + group_effects[group] + np.random.randn(n) * 0.5

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})

        result = lmer("y ~ x + (1 | group)", data, na_action="omit")

        assert result.matrices.n_obs == n
        assert len(result.fitted()) == n

    def test_glmer_na_omit(self) -> None:
        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_effects = np.random.randn(n_groups) * 0.3
        eta = -0.5 + 0.5 * x + group_effects[group]
        p = 1 / (1 + np.exp(-eta))
        y = np.random.binomial(1, p).astype(float)

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        data.loc[0, "y"] = np.nan
        data.loc[5, "x"] = np.nan

        result = glmer("y ~ x + (1 | group)", data, family=families.Binomial(), na_action="omit")

        assert result.matrices.n_obs == n - 2
        assert result.converged

    def test_glmer_na_exclude(self) -> None:
        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_effects = np.random.randn(n_groups) * 0.3
        eta = -0.5 + 0.5 * x + group_effects[group]
        p = 1 / (1 + np.exp(-eta))
        y = np.random.binomial(1, p).astype(float)

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        data.loc[0, "y"] = np.nan
        data.loc[5, "x"] = np.nan

        result = glmer("y ~ x + (1 | group)", data, family=families.Binomial(), na_action="exclude")

        assert result.matrices.n_obs == n - 2

        fitted_vals = result.fitted()
        assert len(fitted_vals) == n
        assert np.isnan(fitted_vals[0])
        assert np.isnan(fitted_vals[5])
        assert not np.isnan(fitted_vals[1])


class TestInfluenceDiagnostics:
    def test_lmer_hatvalues(self) -> None:
        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_effects = np.random.randn(n_groups) * 0.5
        y = 2.0 + 1.5 * x + group_effects[group] + np.random.randn(n) * 0.5

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        result = lmer("y ~ x + (1 | group)", data)

        h = result.hatvalues()

        assert len(h) == n
        assert np.all(h >= 0)
        assert np.all(h < 1)
        assert np.sum(h) > 0

    def test_lmer_cooks_distance(self) -> None:
        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_effects = np.random.randn(n_groups) * 0.5
        y = 2.0 + 1.5 * x + group_effects[group] + np.random.randn(n) * 0.5

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        result = lmer("y ~ x + (1 | group)", data)

        cooks_d = result.cooks_distance()

        assert len(cooks_d) == n
        assert np.all(cooks_d >= 0)
        assert np.all(np.isfinite(cooks_d))

    def test_lmer_influence(self) -> None:
        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_effects = np.random.randn(n_groups) * 0.5
        y = 2.0 + 1.5 * x + group_effects[group] + np.random.randn(n) * 0.5

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        result = lmer("y ~ x + (1 | group)", data)

        infl = result.influence()

        assert "hat" in infl
        assert "cooks_d" in infl
        assert "std_resid" in infl
        assert "student_resid" in infl

        assert len(infl["hat"]) == n
        assert len(infl["cooks_d"]) == n
        assert len(infl["std_resid"]) == n
        assert len(infl["student_resid"]) == n

    def test_lmer_hatvalues_sum_constraint(self) -> None:
        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_effects = np.random.randn(n_groups) * 0.5
        y = 2.0 + 1.5 * x + group_effects[group] + np.random.randn(n) * 0.5

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        result = lmer("y ~ x + (1 | group)", data)

        h = result.hatvalues()

        assert np.sum(h) > 0
        assert np.mean(h) > 0
        assert np.mean(h) < 1

    def test_glmer_hatvalues(self) -> None:
        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_effects = np.random.randn(n_groups) * 0.3
        eta = -0.5 + 0.5 * x + group_effects[group]
        p = 1 / (1 + np.exp(-eta))
        y = np.random.binomial(1, p).astype(float)

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        result = glmer("y ~ x + (1 | group)", data, family=families.Binomial())

        h = result.hatvalues()

        assert len(h) == n
        assert np.all(h >= 0)
        assert np.all(h < 1)

    def test_glmer_cooks_distance(self) -> None:
        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_effects = np.random.randn(n_groups) * 0.3
        eta = -0.5 + 0.5 * x + group_effects[group]
        p = 1 / (1 + np.exp(-eta))
        y = np.random.binomial(1, p).astype(float)

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        result = glmer("y ~ x + (1 | group)", data, family=families.Binomial())

        cooks_d = result.cooks_distance()

        assert len(cooks_d) == n
        assert np.all(cooks_d >= 0)
        assert np.all(np.isfinite(cooks_d))

    def test_glmer_influence(self) -> None:
        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_effects = np.random.randn(n_groups) * 0.3
        eta = -0.5 + 0.5 * x + group_effects[group]
        p = 1 / (1 + np.exp(-eta))
        y = np.random.binomial(1, p).astype(float)

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        result = glmer("y ~ x + (1 | group)", data, family=families.Binomial())

        infl = result.influence()

        assert "hat" in infl
        assert "cooks_d" in infl
        assert "pearson_resid" in infl
        assert "deviance_resid" in infl

        assert len(infl["hat"]) == n
        assert len(infl["cooks_d"]) == n


class TestRePCA:
    def test_repca_basic(self) -> None:
        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_int = np.random.randn(n_groups) * 0.5
        group_slope = np.random.randn(n_groups) * 0.3
        y = 2.0 + 1.5 * x + group_int[group] + group_slope[group] * x + np.random.randn(n) * 0.5

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        result = lmer("y ~ x + (x | group)", data)

        pca = result.rePCA()

        assert "group" in pca.groups
        group_pca = pca["group"]
        assert group_pca.n_terms == 2
        assert len(group_pca.sdev) == 2
        assert len(group_pca.proportion) == 2
        assert len(group_pca.cumulative) == 2

        assert np.all(group_pca.sdev >= 0)
        assert np.all(group_pca.proportion >= 0)
        assert np.all(group_pca.proportion <= 1)
        assert np.isclose(group_pca.cumulative[-1], 1.0, atol=1e-6) or group_pca.cumulative[-1] == 0

    def test_repca_single_term(self) -> None:
        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_effects = np.random.randn(n_groups) * 0.5
        y = 2.0 + 1.5 * x + group_effects[group] + np.random.randn(n) * 0.5

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        result = lmer("y ~ x + (1 | group)", data)

        pca = result.rePCA()

        assert pca["group"].n_terms == 1
        assert len(pca["group"].sdev) == 1
        assert pca["group"].sdev[0] >= 0

    def test_repca_is_singular(self) -> None:
        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_effects = np.random.randn(n_groups) * 0.5
        y = 2.0 + 1.5 * x + group_effects[group] + np.random.randn(n) * 0.5

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        result = lmer("y ~ x + (1 | group)", data)

        pca = result.rePCA()
        singular = pca.is_singular()

        assert isinstance(singular, dict)
        assert "group" in singular
        assert isinstance(singular["group"], (bool, np.bool_))

    def test_repca_str_output(self) -> None:
        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_effects = np.random.randn(n_groups) * 0.5
        y = 2.0 + 1.5 * x + group_effects[group] + np.random.randn(n) * 0.5

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        result = lmer("y ~ x + (x | group)", data)

        pca = result.rePCA()
        output = str(pca)

        assert "Random effect PCA" in output
        assert "group" in output
        assert "PC1" in output
        assert "PC2" in output

    def test_glmer_repca(self) -> None:
        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_effects = np.random.randn(n_groups) * 0.3
        eta = -0.5 + 0.5 * x + group_effects[group]
        p = 1 / (1 + np.exp(-eta))
        y = np.random.binomial(1, p).astype(float)

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        result = glmer("y ~ x + (1 | group)", data, family=families.Binomial())

        pca = result.rePCA()

        assert "group" in pca.groups
        assert pca["group"].n_terms == 1


class TestDotplot:
    def test_dotplot_basic(self) -> None:
        pytest.importorskip("matplotlib")

        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_effects = np.random.randn(n_groups) * 0.5
        y = 2.0 + 1.5 * x + group_effects[group] + np.random.randn(n) * 0.5

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        result = lmer("y ~ x + (1 | group)", data)

        import matplotlib

        matplotlib.use("Agg")

        fig = result.dotplot()
        assert fig is not None

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_dotplot_multiple_terms(self) -> None:
        pytest.importorskip("matplotlib")

        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_int = np.random.randn(n_groups) * 0.5
        group_slope = np.random.randn(n_groups) * 0.3
        y = 2.0 + 1.5 * x + group_int[group] + group_slope[group] * x + np.random.randn(n) * 0.5

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        result = lmer("y ~ x + (x | group)", data)

        import matplotlib

        matplotlib.use("Agg")

        fig = result.dotplot()
        assert fig is not None

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_dotplot_specific_term(self) -> None:
        pytest.importorskip("matplotlib")

        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_int = np.random.randn(n_groups) * 0.5
        group_slope = np.random.randn(n_groups) * 0.3
        y = 2.0 + 1.5 * x + group_int[group] + group_slope[group] * x + np.random.randn(n) * 0.5

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        result = lmer("y ~ x + (x | group)", data)

        import matplotlib

        matplotlib.use("Agg")

        fig = result.dotplot(term="(Intercept)")
        assert fig is not None

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_glmer_dotplot(self) -> None:
        pytest.importorskip("matplotlib")

        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_effects = np.random.randn(n_groups) * 0.3
        eta = -0.5 + 0.5 * x + group_effects[group]
        p = 1 / (1 + np.exp(-eta))
        y = np.random.binomial(1, p).astype(float)

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        result = glmer("y ~ x + (1 | group)", data, family=families.Binomial())

        import matplotlib

        matplotlib.use("Agg")

        fig = result.dotplot()
        assert fig is not None

        import matplotlib.pyplot as plt

        plt.close(fig)


class TestContrasts:
    def test_treatment_contrasts_default(self) -> None:
        np.random.seed(42)
        n = 120
        group = np.repeat(["A", "B", "C", "D"], n // 4)
        subject = np.repeat(np.arange(12), n // 12)
        effects = np.array([0, 1, 2, 3])[np.searchsorted(["A", "B", "C", "D"], group)]
        y = np.random.randn(n) + effects

        data = pd.DataFrame({"y": y, "group": group, "subject": [str(s) for s in subject]})

        result = lmer("y ~ group + (1 | subject)", data)

        fixed_names = result.matrices.fixed_names
        assert "(Intercept)" in fixed_names
        assert any("group" in name for name in fixed_names)
        assert len(fixed_names) == 4

    def test_sum_contrasts(self) -> None:
        np.random.seed(42)
        n = 120
        group = np.repeat(["A", "B", "C", "D"], n // 4)
        subject = np.repeat(np.arange(12), n // 12)
        effects = np.array([0, 1, 2, 3])[np.searchsorted(["A", "B", "C", "D"], group)]
        y = np.random.randn(n) + effects

        data = pd.DataFrame({"y": y, "group": group, "subject": [str(s) for s in subject]})

        result = lmer("y ~ group + (1 | subject)", data, contrasts={"group": "sum"})

        fixed_names = result.matrices.fixed_names
        assert len(fixed_names) == 4
        assert "(Intercept)" in fixed_names

    def test_helmert_contrasts(self) -> None:
        np.random.seed(42)
        n = 120
        group = np.repeat(["A", "B", "C", "D"], n // 4)
        subject = np.repeat(np.arange(12), n // 12)
        effects = np.array([0, 1, 2, 3])[np.searchsorted(["A", "B", "C", "D"], group)]
        y = np.random.randn(n) + effects

        data = pd.DataFrame({"y": y, "group": group, "subject": [str(s) for s in subject]})

        result = lmer("y ~ group + (1 | subject)", data, contrasts={"group": "helmert"})

        fixed_names = result.matrices.fixed_names
        assert len(fixed_names) == 4
        assert result.converged

    def test_poly_contrasts(self) -> None:
        np.random.seed(42)
        n = 120
        group = np.repeat(["A", "B", "C", "D"], n // 4)
        subject = np.repeat(np.arange(12), n // 12)
        effects = np.array([0, 1, 4, 9])[np.searchsorted(["A", "B", "C", "D"], group)]
        y = np.random.randn(n) + effects

        data = pd.DataFrame({"y": y, "group": group, "subject": [str(s) for s in subject]})

        result = lmer("y ~ group + (1 | subject)", data, contrasts={"group": "poly"})

        fixed_names = result.matrices.fixed_names
        assert len(fixed_names) == 4
        assert result.converged

    def test_custom_contrast_matrix(self) -> None:
        np.random.seed(42)
        n = 90
        group = np.repeat(["A", "B", "C"], n // 3)
        subject = np.repeat(np.arange(9), n // 9)
        y = np.random.randn(n) + np.array([0, 1, 2])[np.searchsorted(["A", "B", "C"], group)]

        data = pd.DataFrame({"y": y, "group": group, "subject": [str(s) for s in subject]})

        custom_contrasts = np.array([[-1, -1], [1, 0], [0, 1]], dtype=np.float64)

        result = lmer("y ~ group + (1 | subject)", data, contrasts={"group": custom_contrasts})

        fixed_names = result.matrices.fixed_names
        assert len(fixed_names) == 3
        assert result.converged

    def test_glmer_contrasts(self) -> None:
        np.random.seed(42)
        n = 120
        group = np.repeat(["A", "B", "C"], n // 3)
        subject = np.repeat(np.arange(12), n // 12)
        eta = np.array([-1.0, 0.0, 1.0])[np.searchsorted(["A", "B", "C"], group)]
        p = 1 / (1 + np.exp(-eta))
        y = np.random.binomial(1, p).astype(float)

        data = pd.DataFrame({"y": y, "group": group, "subject": [str(s) for s in subject]})

        result = glmer(
            "y ~ group + (1 | subject)",
            data,
            family=families.Binomial(),
            contrasts={"group": "sum"},
        )

        fixed_names = result.matrices.fixed_names
        assert len(fixed_names) == 3
        assert "(Intercept)" in fixed_names

    def test_contrasts_different_effects(self) -> None:
        from mixedlm.utils.contrasts import contr_sum, contr_treatment

        n = 3
        treatment_matrix = contr_treatment(n)
        sum_matrix = contr_sum(n)

        assert treatment_matrix.shape == (3, 2)
        assert sum_matrix.shape == (3, 2)
        assert not np.allclose(treatment_matrix, sum_matrix)

        assert np.allclose(treatment_matrix[0, :], [0, 0])
        assert np.allclose(sum_matrix[-1, :], [-1, -1])

    def test_contrasts_with_interactions(self) -> None:
        np.random.seed(42)
        n = 240
        group1 = np.tile(np.repeat(["A", "B"], n // 4), 2)
        group2 = np.repeat(["X", "Y"], n // 2)
        subject = np.repeat(np.arange(24), n // 24)
        y = np.random.randn(n)

        data = pd.DataFrame(
            {"y": y, "group1": group1, "group2": group2, "subject": [str(s) for s in subject]}
        )

        result = lmer(
            "y ~ group1 * group2 + (1 | subject)",
            data,
            contrasts={"group1": "sum", "group2": "treatment"},
        )

        assert result.converged
        assert "(Intercept)" in result.matrices.fixed_names
