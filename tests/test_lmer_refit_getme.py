import numpy as np
import pytest
from mixedlm import families, glmer, lmer

from tests._lmer_data import CBPP, SLEEPSTUDY


class TestGetMEComponents:
    def test_getME_X(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        X = result.getME("X")

        assert X.shape == (180, 2)
        assert np.allclose(X[:, 0], 1.0)

    def test_getME_Z(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        Z = result.getME("Z")

        assert Z.shape == (180, 18)

    def test_getME_y(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        y = result.getME("y")

        assert len(y) == 180
        assert np.allclose(y, SLEEPSTUDY["Reaction"].values)

    def test_getME_beta(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        beta = result.getME("beta")

        assert len(beta) == 2
        assert np.allclose(beta, list(result.fixef().values()))

    def test_getME_theta(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        theta = result.getME("theta")

        assert len(theta) == 1
        assert theta[0] >= 0

    def test_getME_Lambda(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        Lambda = result.getME("Lambda")

        assert Lambda.shape == (18, 18)

    def test_getME_u_b(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        u = result.getME("u")
        b = result.getME("b")

        assert len(u) == 18
        assert np.allclose(u, b)

    def test_getME_sigma(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        sigma = result.getME("sigma")

        assert sigma > 0
        assert sigma == result.sigma

    def test_getME_dimensions(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        assert result.getME("n") == 180
        assert result.getME("n_obs") == 180
        assert result.getME("p") == 2
        assert result.getME("n_fixed") == 2
        assert result.getME("q") == 18
        assert result.getME("n_random") == 18

    def test_getME_lower(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        lower = result.getME("lower")

        assert len(lower) == 1
        assert lower[0] == 0.0

    def test_getME_lower_correlated(self) -> None:
        result = lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY)
        lower = result.getME("lower")

        assert len(lower) == 3
        assert lower[0] == 0.0
        assert lower[1] == -np.inf
        assert lower[2] == 0.0

    def test_getME_weights_offset(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        weights = result.getME("weights")
        assert len(weights) == 180
        assert np.allclose(weights, 1.0)

        offset = result.getME("offset")
        assert len(offset) == 180
        assert np.allclose(offset, 0.0)

    def test_getME_REML_deviance(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        assert result.getME("REML") is True
        assert result.getME("deviance") > 0

    def test_getME_flist_cnms(self) -> None:
        result = lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY)

        flist = result.getME("flist")
        assert flist == ["Subject"]

        cnms = result.getME("cnms")
        assert "Subject" in cnms
        assert "(Intercept)" in cnms["Subject"]
        assert "Days" in cnms["Subject"]

    def test_getME_Gp(self) -> None:
        result = lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY)
        Gp = result.getME("Gp")

        assert len(Gp) == 2
        assert Gp[0] == 0
        assert Gp[1] == 36

    def test_getME_invalid(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        with pytest.raises(ValueError, match="Unknown component name"):
            result.getME("invalid_name")

    def test_getME_glmer(self) -> None:
        data = CBPP.copy()
        data["y"] = data["incidence"] / data["size"]

        result = glmer("y ~ period + (1 | herd)", data, family=families.Binomial())

        X = result.getME("X")
        assert X.shape[1] == 4

        family = result.getME("family")
        assert isinstance(family, families.Binomial)


class TestUpdateSleepstudy:
    def test_update_REML(self) -> None:
        result1 = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        assert result1.REML is True

        result2 = result1.update(REML=False)
        assert result2.REML is False

    def test_update_same_formula(self) -> None:
        result1 = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        result2 = result1.update()

        fixef1 = np.array(list(result1.fixef().values()))
        fixef2 = np.array(list(result2.fixef().values()))
        assert np.allclose(fixef1, fixef2, rtol=1e-4)

    def test_update_new_formula(self) -> None:
        data = SLEEPSTUDY.copy()
        data["Days2"] = data["Days"] ** 2

        result1 = lmer("Reaction ~ Days + (1 | Subject)", data)
        result2 = result1.update("Reaction ~ Days + Days2 + (1 | Subject)", data=data)

        assert len(result1.fixef()) == 2
        assert len(result2.fixef()) == 3

    def test_update_new_data(self) -> None:
        result1 = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        subset = SLEEPSTUDY[SLEEPSTUDY["Days"] <= 5].copy()
        result2 = result1.update(data=subset)

        assert result2.getME("n") < result1.getME("n")

    def test_update_glmer_family(self) -> None:
        data = CBPP.copy()
        data["y"] = data["incidence"] / data["size"]

        result1 = glmer("y ~ period + (1 | herd)", data, family=families.Binomial())

        data["count"] = np.round(data["incidence"]).astype(int)
        result2 = result1.update(
            formula="count ~ period + (1 | herd)", data=data, family=families.Poisson()
        )

        assert isinstance(result1.getME("family"), families.Binomial)
        assert isinstance(result2.getME("family"), families.Poisson)

    def test_update_with_weights(self) -> None:
        result1 = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        weights = np.ones(180)
        weights[:90] = 2.0
        result2 = result1.update(weights=weights)

        w1 = result1.getME("weights")
        w2 = result2.getME("weights")

        assert np.allclose(w1, 1.0)
        assert np.allclose(w2[:90], 2.0)

    def test_update_formula_dot_syntax_add(self) -> None:
        data = SLEEPSTUDY.copy()
        data["Days2"] = data["Days"] ** 2

        result1 = lmer("Reaction ~ Days + (1 | Subject)", data)
        result2 = result1.update(". ~ . + Days2", data=data)

        assert len(result1.fixef()) == 2
        assert len(result2.fixef()) == 3


class TestRefitSleepstudy:
    def test_refit_same_response(self) -> None:
        result1 = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        newresp = SLEEPSTUDY["Reaction"].values.copy()
        result2 = result1.refit(newresp)

        fixef1 = np.array(list(result1.fixef().values()))
        fixef2 = np.array(list(result2.fixef().values()))
        assert np.allclose(fixef1, fixef2, rtol=1e-4)

    def test_refit_new_response(self) -> None:
        result1 = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        np.random.seed(42)
        newresp = SLEEPSTUDY["Reaction"].values + np.random.normal(0, 50, 180)
        result2 = result1.refit(newresp)

        fixef1 = np.array(list(result1.fixef().values()))
        fixef2 = np.array(list(result2.fixef().values()))
        assert not np.allclose(fixef1, fixef2, rtol=1e-4)

    def test_refit_wrong_size(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        with pytest.raises(ValueError, match="length"):
            result.refit(np.array([1.0, 2.0, 3.0]))

    def test_refit_preserves_structure(self) -> None:
        result1 = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        newresp = SLEEPSTUDY["Reaction"].values + 100
        result2 = result1.refit(newresp)

        assert result1.getME("n") == result2.getME("n")
        assert result1.getME("p") == result2.getME("p")
        assert result1.getME("q") == result2.getME("q")

    def test_refit_multiple_times(self) -> None:
        result1 = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        newresp = SLEEPSTUDY["Reaction"].values.copy()
        result2 = result1.refit(newresp)
        result3 = result2.refit(newresp)

        fixef2 = np.array(list(result2.fixef().values()))
        fixef3 = np.array(list(result3.fixef().values()))
        assert np.allclose(fixef2, fixef3, rtol=1e-4)

    def test_refit_glmer_basic(self) -> None:
        data = CBPP.copy()
        data["y"] = data["incidence"] / data["size"]

        result1 = glmer("y ~ period + (1 | herd)", data, family=families.Binomial())

        np.random.seed(42)
        newresp = np.clip(data["y"].values + np.random.normal(0, 0.1, len(data)), 0.01, 0.99)
        result2 = result1.refit(newresp)

        assert result1.getME("n") == result2.getME("n")

    def test_refit_glmer_wrong_size(self) -> None:
        data = CBPP.copy()
        data["y"] = data["incidence"] / data["size"]

        result = glmer("y ~ period + (1 | herd)", data, family=families.Binomial())

        with pytest.raises(ValueError, match="length"):
            result.refit(np.array([0.1, 0.2, 0.3]))

    def test_refit_simulation_workflow(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        np.random.seed(123)
        fixef_samples = []
        for _ in range(3):
            newresp = SLEEPSTUDY["Reaction"].values + np.random.normal(0, 30, 180)
            refit_result = result.refit(newresp)
            fixef_samples.append(list(refit_result.fixef().values()))

        fixef_array = np.array(fixef_samples)
        assert fixef_array.shape == (3, 2)


class TestRefitML:
    def test_refitML_basic(self) -> None:
        result_reml = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, REML=True)
        assert result_reml.REML is True

        result_ml = result_reml.refitML()
        assert result_ml.REML is False

    def test_refitML_preserves_structure(self) -> None:
        result_reml = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, REML=True)
        result_ml = result_reml.refitML()

        assert result_reml.getME("n") == result_ml.getME("n")
        assert result_reml.getME("p") == result_ml.getME("p")
        assert result_reml.getME("q") == result_ml.getME("q")

    def test_refitML_different_estimates(self) -> None:
        result_reml = lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY, REML=True)
        result_ml = result_reml.refitML()

        theta_reml = result_reml.theta
        theta_ml = result_ml.theta

        assert result_reml.REML is True
        assert result_ml.REML is False
        assert theta_reml.shape == theta_ml.shape

    def test_refitML_already_ML_returns_self(self) -> None:
        result_ml = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, REML=False)
        result_ml2 = result_ml.refitML()

        assert result_ml is result_ml2

    def test_refitML_for_LRT(self) -> None:
        data = SLEEPSTUDY.copy()
        data["Days2"] = data["Days"] ** 2

        result_full = lmer("Reaction ~ Days + Days2 + (1 | Subject)", data, REML=True)
        result_reduced = lmer("Reaction ~ Days + (1 | Subject)", data, REML=True)

        ml_full = result_full.refitML()
        ml_reduced = result_reduced.refitML()

        ll_full = ml_full.logLik().value
        ll_reduced = ml_reduced.logLik().value

        assert ll_full > ll_reduced
