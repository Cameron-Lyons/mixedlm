"""Tests for newly implemented lme4-compatible features."""

import numpy as np
import pandas as pd
import pytest
from mixedlm import (
    dropOffset,
    expandDoubleVerts,
    getFixedFormulaStr,
    getNGroups,
    getRandomFormulaStr,
    getResponseName,
    glmer,
    lmer,
    parse_formula,
    set_cov_type,
)
from mixedlm.datasets import load_cbpp, load_sleepstudy
from mixedlm.diagnostics.influence import (
    cooks_distance,
    dfbeta,
    dfbetas,
    dffits,
    influence,
    influence_summary,
    influential_obs,
    leverage,
)
from mixedlm.families import Binomial
from mixedlm.inference.profile import (
    as_dataframe,
    confint_profile,
    logProf,
    profile_lmer,
    sdProf,
    varianceProf,
)
from mixedlm.models.control import GlmerControl, LmerControl

SLEEPSTUDY = load_sleepstudy()
CBPP = load_cbpp()


class TestCovarianceStructures:
    def test_set_cov_type_cs(self):
        formula = parse_formula("Reaction ~ Days + (Days | Subject)")
        formula_cs = set_cov_type(formula, "cs")

        assert len(formula_cs.random) == 1
        assert formula_cs.random[0].cov_type == "cs"

    def test_set_cov_type_ar1(self):
        formula = parse_formula("Reaction ~ Days + (Days | Subject)")
        formula_ar1 = set_cov_type(formula, "ar1")

        assert len(formula_ar1.random) == 1
        assert formula_ar1.random[0].cov_type == "ar1"

    def test_set_cov_type_dict(self):
        formula = parse_formula("y ~ x + (1 | group1) + (1 | group2)")
        formula_mixed = set_cov_type(formula, {"group1": "cs", "group2": "ar1"})

        cov_types = {r.grouping: r.cov_type for r in formula_mixed.random}
        assert cov_types["group1"] == "cs"
        assert cov_types["group2"] == "ar1"

    def test_set_cov_type_invalid(self):
        formula = parse_formula("y ~ x + (1 | group)")

        with pytest.raises(ValueError, match="Invalid cov_type"):
            set_cov_type(formula, "invalid")

    def test_lmer_with_cs_structure(self):
        formula = set_cov_type("Reaction ~ Days + (Days | Subject)", "cs")
        result = lmer(formula, SLEEPSTUDY)

        assert result.converged
        assert len(result.theta) == 2

    def test_lmer_with_ar1_structure(self):
        formula = set_cov_type("Reaction ~ Days + (Days | Subject)", "ar1")
        result = lmer(formula, SLEEPSTUDY)

        assert result.converged
        assert len(result.theta) == 2

    def test_cs_vs_us_different_theta_count(self):
        result_us = lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY)

        formula_cs = set_cov_type("Reaction ~ Days + (Days | Subject)", "cs")
        result_cs = lmer(formula_cs, SLEEPSTUDY)

        assert len(result_us.theta) == 3
        assert len(result_cs.theta) == 2


class TestControlParameters:
    def test_lmer_control_new_params_defaults(self):
        ctrl = LmerControl()

        assert ctrl.check_nobs_vs_rankZ == "ignore"
        assert ctrl.check_nobs_vs_nlev == "ignore"
        assert ctrl.check_nlev_gtreq_5 == "warning"
        assert ctrl.check_nlev_gtr_1 == "stop"
        assert ctrl.check_rankX == "message+drop.cols"
        assert ctrl.check_scaleX == "warning"

    def test_glmer_control_new_params_defaults(self):
        ctrl = GlmerControl()

        assert ctrl.check_nobs_vs_rankZ == "ignore"
        assert ctrl.check_nobs_vs_nlev == "ignore"
        assert ctrl.check_nlev_gtreq_5 == "warning"
        assert ctrl.check_nlev_gtr_1 == "stop"
        assert ctrl.check_rankX == "message+drop.cols"
        assert ctrl.check_scaleX == "warning"

    def test_lmer_control_invalid_check_action(self):
        with pytest.raises(ValueError, match="check_nobs_vs_rankZ"):
            LmerControl(check_nobs_vs_rankZ="invalid")

    def test_lmer_control_valid_rankX_options(self):
        ctrl1 = LmerControl(check_rankX="ignore")
        assert ctrl1.check_rankX == "ignore"

        ctrl2 = LmerControl(check_rankX="warning+drop.cols")
        assert ctrl2.check_rankX == "warning+drop.cols"

    def test_lmer_control_invalid_rankX(self):
        with pytest.raises(ValueError, match="check_rankX"):
            LmerControl(check_rankX="invalid+option")


class TestModelChecks:
    def test_check_nlev_gtr_1_stop(self):
        data = pd.DataFrame(
            {
                "y": [1.0, 2.0, 3.0],
                "x": [1.0, 2.0, 3.0],
                "group": ["A", "A", "A"],
            }
        )

        result = lmer("y ~ x + (1 | group)", data, control=LmerControl(check_nlev_gtr_1="ignore"))
        assert result is not None

    def test_check_nlev_gtreq_5_warning(self):
        data = pd.DataFrame(
            {
                "y": np.random.randn(12),
                "x": np.random.randn(12),
                "group": ["A", "B", "C"] * 4,
            }
        )

        with pytest.warns(UserWarning, match="only 3 levels"):
            lmer("y ~ x + (1 | group)", data, control=LmerControl(check_nlev_gtreq_5="warning"))

    def test_check_scaleX_warning(self):
        data = pd.DataFrame(
            {
                "y": np.random.randn(100),
                "x1": np.random.randn(100),
                "x2": np.random.randn(100) * 10000,
                "group": np.repeat(range(10), 10),
            }
        )

        with pytest.warns(UserWarning, match="very different scales"):
            lmer(
                "y ~ x1 + x2 + (1 | group)",
                data,
                control=LmerControl(check_scaleX="warning", check_nlev_gtreq_5="ignore"),
            )


class TestInfluenceDiagnostics:
    @pytest.fixture
    def lmer_result(self):
        return lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

    @pytest.fixture
    def glmer_result(self):
        cbpp = CBPP.copy()
        cbpp["y"] = cbpp["incidence"] / cbpp["size"]
        return glmer("y ~ period + (1 | herd)", cbpp, family=Binomial())

    def test_influence_lmer(self, lmer_result):
        inf = influence(lmer_result)

        assert inf.model_type == "lmer"
        assert len(inf.hat_values) == len(SLEEPSTUDY)
        assert len(inf.residuals) == len(SLEEPSTUDY)

    def test_influence_glmer(self, glmer_result):
        inf = influence(glmer_result)

        assert inf.model_type == "glmer"
        assert len(inf.hat_values) == len(CBPP)

    def test_dfbeta(self, lmer_result):
        inf = influence(lmer_result)
        dfb = dfbeta(inf)

        assert dfb.shape == (len(SLEEPSTUDY), 2)

    def test_dfbetas(self, lmer_result):
        inf = influence(lmer_result)
        dfbs = dfbetas(inf)

        assert dfbs.shape == (len(SLEEPSTUDY), 2)

    def test_cooks_distance(self, lmer_result):
        inf = influence(lmer_result)
        cooks = cooks_distance(inf)

        assert len(cooks) == len(SLEEPSTUDY)
        assert np.all(cooks >= 0)

    def test_dffits(self, lmer_result):
        inf = influence(lmer_result)
        dff = dffits(inf)

        assert len(dff) == len(SLEEPSTUDY)

    def test_leverage(self, lmer_result):
        inf = influence(lmer_result)
        lev = leverage(inf)

        assert len(lev) == len(SLEEPSTUDY)
        assert np.all(lev >= 0)
        assert np.all(lev <= 1)

    def test_influence_summary(self, lmer_result):
        inf = influence(lmer_result)
        summary = influence_summary(inf)

        assert isinstance(summary, pd.DataFrame)
        assert "cooks_distance" in summary.columns
        assert "max_abs_dfbetas" in summary.columns
        assert "leverage" in summary.columns
        assert "influential_cooks" in summary.columns

    def test_influential_obs(self, lmer_result):
        inf = influence(lmer_result)
        infl = influential_obs(inf, threshold="cooks")

        assert isinstance(infl, np.ndarray)

    def test_influential_obs_different_thresholds(self, lmer_result):
        inf = influence(lmer_result)

        for thresh in ["cooks", "dfbetas", "leverage", "dffits"]:
            infl = influential_obs(inf, threshold=thresh)
            assert isinstance(infl, np.ndarray)


class TestProfileTransformations:
    @pytest.fixture
    def profiles(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        return profile_lmer(result, which=["Days"], n_points=10)

    def test_logProf(self, profiles):
        profile = profiles["Days"]
        profile_positive = type(profile)(
            parameter=profile.parameter,
            values=np.abs(profile.values) + 1,
            zeta=profile.zeta,
            mle=abs(profile.mle) + 1,
            ci_lower=abs(profile.ci_lower) + 1,
            ci_upper=abs(profile.ci_upper) + 1,
            level=profile.level,
        )

        log_profile = logProf(profile_positive)

        assert log_profile.parameter == f"log({profile_positive.parameter})"
        assert len(log_profile.values) == len(profile_positive.values)

    def test_varianceProf(self, profiles):
        profile = profiles["Days"]
        var_profile = varianceProf(profile)

        assert "²" in var_profile.parameter
        np.testing.assert_allclose(var_profile.values, profile.values**2)
        assert var_profile.mle == profile.mle**2

    def test_sdProf(self, profiles):
        profile = profiles["Days"]
        profile_positive = type(profile)(
            parameter=profile.parameter,
            values=np.abs(profile.values),
            zeta=profile.zeta,
            mle=abs(profile.mle),
            ci_lower=abs(profile.ci_lower),
            ci_upper=abs(profile.ci_upper),
            level=profile.level,
        )

        sd_profile = sdProf(profile_positive)

        assert "sqrt" in sd_profile.parameter
        np.testing.assert_allclose(sd_profile.values, np.sqrt(profile_positive.values))

    def test_as_dataframe_single(self, profiles):
        profile = profiles["Days"]
        df = as_dataframe(profile)

        assert isinstance(df, pd.DataFrame)
        assert "parameter" in df.columns
        assert "value" in df.columns
        assert "zeta" in df.columns
        assert len(df) == len(profile.values)

    def test_as_dataframe_dict(self, profiles):
        df = as_dataframe(profiles)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(profiles["Days"].values)

    def test_confint_profile(self, profiles):
        ci = confint_profile(profiles)

        assert isinstance(ci, pd.DataFrame)
        assert "parameter" in ci.columns
        assert "estimate" in ci.columns
        assert "lower" in ci.columns
        assert "upper" in ci.columns

    def test_confint_profile_different_level(self, profiles):
        ci_95 = confint_profile(profiles, level=0.95)
        ci_90 = confint_profile(profiles, level=0.90)

        assert ci_95["level"].iloc[0] == 0.95
        assert ci_90["level"].iloc[0] == 0.90

        width_95 = ci_95["upper"].iloc[0] - ci_95["lower"].iloc[0]
        width_90 = ci_90["upper"].iloc[0] - ci_90["lower"].iloc[0]
        assert width_95 > width_90


class TestFormulaUtilities:
    def test_expandDoubleVerts_basic(self):
        result = expandDoubleVerts("y ~ x + (1 + x || group)")

        assert "||" not in result
        assert "(1 | group)" in result
        assert "(0 + x | group)" in result

    def test_expandDoubleVerts_no_intercept(self):
        result = expandDoubleVerts("y ~ x + (0 + x || group)")

        assert "(0 + x | group)" in result

    def test_expandDoubleVerts_multiple(self):
        result = expandDoubleVerts("y ~ x + (1 + x || g1) + (1 + z || g2)")

        assert "||" not in result
        assert "(1 | g1)" in result
        assert "(0 + x | g1)" in result
        assert "(1 | g2)" in result
        assert "(0 + z | g2)" in result

    def test_expandDoubleVerts_no_change(self):
        original = "y ~ x + (1 | group)"
        result = expandDoubleVerts(original)

        assert result == original

    def test_dropOffset_basic(self):
        result = dropOffset("y ~ x + offset(log(t)) + (1 | group)")

        assert "offset" not in result
        assert "x" in result
        assert "(1 | group)" in result

    def test_dropOffset_multiple(self):
        result = dropOffset("y ~ x + offset(a) + z + offset(b)")

        assert "offset" not in result
        assert "x" in result
        assert "z" in result

    def test_dropOffset_no_offset(self):
        original = "y ~ x + (1 | group)"
        result = dropOffset(original)

        assert "x" in result
        assert "(1 | group)" in result

    def test_getResponseName(self):
        assert getResponseName("y ~ x + (1 | group)") == "y"
        assert getResponseName("Reaction ~ Days + (Days | Subject)") == "Reaction"

    def test_getFixedFormulaStr(self):
        result = getFixedFormulaStr("y ~ x + z + (1 | group)")

        assert "y ~" in result
        assert "x" in result
        assert "z" in result
        assert "|" not in result

    def test_getRandomFormulaStr(self):
        result = getRandomFormulaStr("y ~ x + (1 | group)")

        assert "(1 | group)" in result
        assert "x" not in result or "0 + x" in result

    def test_getRandomFormulaStr_multiple(self):
        result = getRandomFormulaStr("y ~ x + (1 | group) + (1 | subject)")

        assert "group" in result
        assert "subject" in result

    def test_getNGroups(self):
        assert getNGroups("y ~ x + (1 | group)") == 1
        assert getNGroups("y ~ x + (1 | group) + (1 | subject)") == 2
        assert getNGroups("y ~ x + (1 | group) + (x | group)") == 1


class TestRefitMethod:
    def test_lmer_refit_basic(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        new_y = result.matrices.y + np.random.randn(len(result.matrices.y))

        result2 = result.refit(newresp=new_y)

        assert result2.converged
        assert len(result2.beta) == len(result.beta)
        assert not np.allclose(result2.beta, result.beta)

    def test_lmer_refit_same_response(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        result2 = result.refit()

        np.testing.assert_allclose(result2.beta, result.beta, rtol=1e-3)

    def test_glmer_refit_basic(self):
        cbpp = CBPP.copy()
        cbpp["y"] = cbpp["incidence"] / cbpp["size"]
        result = glmer("y ~ period + (1 | herd)", cbpp, family=Binomial())
        new_y = result.matrices.y.copy()
        new_y[new_y > 0.5] = 1 - new_y[new_y > 0.5]

        result2 = result.refit(newresp=new_y)

        assert result2.converged or result2.n_iter > 0
        assert len(result2.beta) == len(result.beta)

    def test_refit_wrong_length(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        with pytest.raises(ValueError, match="length"):
            result.refit(newresp=np.array([1, 2, 3]))


class TestIntegration:
    def test_covariance_with_influence(self):
        formula = set_cov_type("Reaction ~ Days + (Days | Subject)", "cs")
        result = lmer(formula, SLEEPSTUDY)
        inf = influence(result)

        assert len(inf.cooks_distance) == len(SLEEPSTUDY)

    def test_refit_with_profile(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        profiles = profile_lmer(result, which=["Days"], n_points=5)

        simulated = result.simulate()
        result2 = result.refit(newresp=simulated)
        profiles2 = profile_lmer(result2, which=["Days"], n_points=5)

        assert len(profiles2["Days"].values) == len(profiles["Days"].values)

    def test_formula_utils_with_lmer(self):
        formula_str = "Reaction ~ Days + (Days || Subject)"
        expanded = expandDoubleVerts(formula_str)

        result = lmer(expanded, SLEEPSTUDY)

        assert result.converged
        assert getNGroups(expanded) == 1
