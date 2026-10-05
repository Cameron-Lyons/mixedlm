from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from mixedlm import coef, fixef, getME, nlme, nlmer, ranef
from mixedlm.inference.bootstrap import bootstrap_nlmer
from mixedlm.models.nlmer import NlmerResult

from tests.test_reporting import nlmm_model as nlmm_model


def create_nlme_data(n_groups: int = 8, n_per_group: int = 10, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    data_rows = []
    for subj in range(n_groups):
        asym = 200 + rng.standard_normal() * 20
        r0 = 180 + rng.standard_normal() * 10
        lrc = -3 + rng.standard_normal() * 0.2
        for t in np.linspace(0, 10, n_per_group):
            y = asym + (r0 - asym) * np.exp(-np.exp(lrc) * t) + rng.standard_normal() * 5
            data_rows.append({"subject": f"S{subj + 1}", "time": t, "y": y})
    return pd.DataFrame(data_rows)


def create_offset_nlme_data(seed: int = 20260803) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data_rows = []
    for subject in range(8):
        asym = 200 + rng.normal(0, 12)
        r0 = 180 + rng.normal(0, 6)
        lrc = -3 + rng.normal(0, 0.1)
        for time in np.linspace(0, 10, 10):
            y = asym + (r0 - asym) * np.exp(-np.exp(lrc) * time) + rng.normal(0, 2)
            data_rows.append({"subject": f"S{subject + 1}", "time": time, "y": y})
    return pd.DataFrame(data_rows)


NLME_DATA = create_nlme_data()


def fit_nlme(**kwargs) -> NlmerResult:
    """Fit a random asymptote, which NLME_DATA identifies well.

    With all three parameters random this data is ill-conditioned: one-ulp
    changes to the response decide whether the fit converges.
    """
    return nlmer(
        nlme.SSasymp(),
        NLME_DATA,
        x_var="time",
        y_var="y",
        group_var="subject",
        random_params=["Asym"],
        **kwargs,
    )


def create_logistic_growth_data(seed: int = 7) -> pd.DataFrame:
    """Simulate Orange-like growth with correlated random asymptotes and midpoints."""
    rng = np.random.default_rng(seed)
    ages = np.array([100.0, 250.0, 400.0, 550.0, 700.0, 850.0, 1000.0, 1200.0, 1400.0, 1600.0])
    # Standard deviations 25 and 60 with correlation 0.4.
    covariance = np.array([[625.0, 600.0], [600.0, 3600.0]])
    effects = rng.multivariate_normal([0.0, 0.0], covariance, size=12)
    rows = []
    for tree, (asym_effect, xmid_effect) in enumerate(effects):
        mean = (200.0 + asym_effect) / (1.0 + np.exp((700.0 + xmid_effect - ages) / 350.0))
        noisy = mean + rng.normal(0.0, 4.0, len(ages))
        rows.extend(
            {"tree": f"T{tree + 1:02d}", "age": age, "circumference": value}
            for age, value in zip(ages, noisy, strict=True)
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def growth_fit() -> NlmerResult:
    """Two correlated random effects; refits from perturbed or simulated data converge."""
    result = nlmer(
        nlme.SSlogis(),
        create_logistic_growth_data(),
        x_var="age",
        y_var="circumference",
        group_var="tree",
        random_params=["Asym", "xmid"],
    )
    assert result.converged and result.pnls_converged
    assert len(result.theta) == 3
    return result


def _conditional_jacobian(result: NlmerResult) -> np.ndarray:
    """Analytic derivatives of the conditional mean with respect to phi."""
    jacobian = np.empty((len(result.y), len(result.phi)))
    for group in range(len(result.group_levels)):
        rows = result.groups == group
        params = result.phi.copy()
        params[result.random_params] += result.b[group]
        jacobian[rows] = result.model.gradient(params, result.x[rows])
    return jacobian


class TestNlmerMultipleRandomEffects:
    def test_ranef_coef_and_getme_share_the_random_effects(self, growth_fit) -> None:
        b = growth_fit.getME("b")
        ranefs = ranef(growth_fit)["tree"]
        coefs = coef(growth_fit)["tree"]

        assert b.shape == (12, 2)
        assert list(ranefs) == ["Asym", "xmid"]
        np.testing.assert_array_equal(ranefs["Asym"], b[:, 0])
        np.testing.assert_array_equal(ranefs["xmid"], b[:, 1])
        np.testing.assert_allclose(coefs["Asym"], growth_fit.phi[0] + b[:, 0])
        np.testing.assert_allclose(coefs["xmid"], growth_fit.phi[1] + b[:, 1])
        np.testing.assert_allclose(coefs["scal"], growth_fit.phi[2])

        first_tree = growth_fit.groups == 0
        params = np.array([coefs["Asym"][0], coefs["xmid"][0], coefs["scal"][0]])
        expected = growth_fit.model.predict(params, growth_fit.x[first_tree])
        np.testing.assert_allclose(growth_fit.fitted()[first_tree], expected)

    def test_simulate_draws_correlated_effects_within_trees(self, growth_fit) -> None:
        simulations = growth_fit.simulate(nsim=2000, seed=1)
        correlation = np.corrcoef(simulations - simulations.mean(axis=1, keepdims=True))

        np.testing.assert_array_equal(simulations, growth_fit.simulate(nsim=2000, seed=1))
        # The last two ages of a tree share its asymptote; different trees do not.
        assert correlation[8, 9] > 0.9
        assert abs(correlation[9, 19]) < 0.1

    def test_simulate_without_random_effects_adds_only_residual_noise(self, growth_fit) -> None:
        simulations = growth_fit.simulate(nsim=2000, seed=1, use_re=False)
        population = growth_fit._conditional_mean(random_effects=np.zeros_like(growth_fit.b))
        standardized = (simulations - population[:, None]) / growth_fit.sigma

        assert abs(standardized.mean()) < 0.01
        assert standardized.std() == pytest.approx(1.0, rel=0.02)

    def test_refit_and_update_reproduce_the_fit(self, growth_fit) -> None:
        refitted = growth_fit.refit()
        updated = growth_fit.update()

        assert refitted.converged and refitted.pnls_converged
        np.testing.assert_allclose(refitted.phi, growth_fit.phi, rtol=1e-8)
        np.testing.assert_allclose(refitted.theta, growth_fit.theta, atol=1e-5)
        assert refitted.deviance == pytest.approx(growth_fit.deviance, abs=1e-6)
        for field in ("phi", "theta", "b", "deviance"):
            np.testing.assert_array_equal(getattr(updated, field), getattr(growth_fit, field))

    def test_refit_converges_on_simulated_responses(self, growth_fit) -> None:
        for seed in range(3):
            simulated = growth_fit.simulate(seed=seed)
            refitted = growth_fit.refit(simulated)

            assert refitted.converged and refitted.pnls_converged
            np.testing.assert_array_equal(refitted.y, simulated)
            assert refitted.b.shape == growth_fit.b.shape

    def test_vcov_matches_the_gauss_newton_information(self, growth_fit) -> None:
        jacobian = _conditional_jacobian(growth_fit)
        expected = growth_fit.sigma**2 * np.linalg.inv(jacobian.T @ jacobian)

        np.testing.assert_allclose(growth_fit.vcov(), expected, rtol=0.02)

    def test_hatvalues_project_onto_the_conditional_jacobian(self, growth_fit) -> None:
        jacobian = _conditional_jacobian(growth_fit)
        expected = np.einsum(
            "ij,ji->i", jacobian, np.linalg.solve(jacobian.T @ jacobian, jacobian.T)
        )
        hat = growth_fit.hatvalues()

        np.testing.assert_allclose(hat, expected, atol=1e-6)
        assert hat.sum() == pytest.approx(len(growth_fit.phi), abs=1e-6)

    def test_cooks_distance_and_influence_use_the_hat_values(self, growth_fit) -> None:
        hat = growth_fit.hatvalues()
        pearson = growth_fit.residuals("pearson")
        expected = pearson**2 / len(growth_fit.phi) * hat / (1 - hat) ** 2
        influence = growth_fit.influence()

        np.testing.assert_allclose(growth_fit.cooks_distance(), expected, rtol=1e-12)
        np.testing.assert_allclose(influence["cooks_d"], expected, rtol=1e-12)
        np.testing.assert_allclose(influence["hat"], hat)
        np.testing.assert_allclose(influence["std_resid"], pearson / np.sqrt(1 - hat))

    def test_is_singular_uses_the_covariance_not_the_cholesky_entries(self, growth_fit) -> None:
        assert not growth_fit.isSingular()
        # Independent effects have a zero off-diagonal Cholesky entry.
        assert not replace(growth_fit, theta=np.array([1.0, 0.0, 1.0])).isSingular()
        # Perfectly correlated effects have a rank-one covariance.
        assert replace(growth_fit, theta=np.array([1.0, 1.0, 0.0])).isSingular()
        assert replace(growth_fit, theta=np.array([0.0, 0.0, 1.0])).is_singular()
        with pytest.raises(ValueError, match="tol must be"):
            growth_fit.isSingular(tol=-1.0)


class TestNlmerPredict:
    def test_grouped_prediction_batches_rows_and_preserves_order(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(
            model,
            NLME_DATA,
            x_var="time",
            y_var="y",
            group_var="subject",
            random_params=["Asym", "R0"],
        )
        first_group, second_group = result.group_levels[:2]
        new_data = pd.DataFrame(
            {
                "time": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
                "subject": [first_group, "new-a", second_group, first_group, "new-b", second_group],
            }
        )

        expected = np.empty(len(new_data), dtype=np.float64)
        group_lookup = {group: index for index, group in enumerate(result.group_levels)}
        for row, (x_value, group) in enumerate(
            zip(new_data["time"], new_data["subject"], strict=True)
        ):
            params = result.phi.copy()
            group_index = group_lookup.get(group)
            if group_index is not None:
                params[result.random_params] += result.b[group_index]
            expected[row] = model.predict(params, np.array([x_value]))[0]

        with patch.object(model, "predict", wraps=model.predict) as predict:
            actual = result.predict(new_data, group_var="subject")

        assert np.allclose(actual, expected)
        assert predict.call_count == 3


class TestNlmerSimulate:
    def test_simulate_single(self) -> None:
        result = fit_nlme()

        sim = result.simulate(nsim=1, seed=123)
        assert sim.shape == (len(NLME_DATA),)
        assert not np.allclose(sim, result.y)

    def test_simulate_multiple(self) -> None:
        result = fit_nlme()

        sim = result.simulate(nsim=5, seed=123)
        assert sim.shape == (len(NLME_DATA), 5)

    def test_simulate_reproducible(self) -> None:
        result = fit_nlme()

        sim1 = result.simulate(nsim=1, seed=42)
        sim2 = result.simulate(nsim=1, seed=42)
        assert np.allclose(sim1, sim2)

    def test_simulate_no_random_effects(self) -> None:
        result = fit_nlme()

        sim_with_re = result.simulate(nsim=1, seed=123, use_re=True)
        sim_no_re = result.simulate(nsim=1, seed=123, use_re=False)
        assert not np.allclose(sim_with_re, sim_no_re)

    def test_simulate_re_form_na(self) -> None:
        result = fit_nlme()

        sim = result.simulate(nsim=1, seed=123, re_form="NA")
        assert sim.shape == (len(NLME_DATA),)

    def test_simulate_uses_inverse_weight_residual_variance(self) -> None:
        weights = np.ones(len(NLME_DATA))
        weights[1] = 4.0
        result = fit_nlme(weights=weights)

        simulations = result.simulate(nsim=1500, seed=123, use_re=False)
        empirical_scale = np.std(simulations, axis=1)

        assert empirical_scale[0] / empirical_scale[1] == pytest.approx(2.0, rel=0.1)


class TestNlmerRefit:
    def test_refit_same_response(self) -> None:
        result = fit_nlme()

        refit_result = result.refit()
        assert len(refit_result.phi) == len(result.phi)
        assert refit_result.deviance is not None
        if refit_result.converged and not np.any(np.isnan(refit_result.phi)):
            assert np.allclose(result.phi, refit_result.phi, atol=1.0)

    def test_refit_new_response(self) -> None:
        result = fit_nlme()

        new_y = result.simulate(nsim=1, seed=456)
        refit_result = result.refit(new_y)

        assert len(refit_result.phi) == len(result.phi)
        assert np.allclose(refit_result.y, new_y)

    def test_refit_wrong_length_raises(self) -> None:
        result = fit_nlme()

        with pytest.raises(ValueError, match="newresp has length"):
            result.refit(np.array([1, 2, 3]))


class TestNlmerUpdate:
    def test_update_same_data(self) -> None:
        result = fit_nlme()

        updated = result.update()
        for field in ("phi", "theta", "b", "deviance", "converged", "pnls_converged"):
            np.testing.assert_array_equal(getattr(updated, field), getattr(result, field))

    def test_update_with_start(self, request) -> None:
        result = request.getfixturevalue("nlmm_model")
        start = {"Asym": 11.0, "R0": 3.0, "lrc": -0.8}
        updated = result.update(start=start)
        assert updated.converged and updated.pnls_converged
        direct = nlmer(
            result.model,
            result.model_frame(),
            x_var="time",
            y_var="response",
            group_var="subject",
            start=start,
        )
        for field in ("phi", "theta", "b", "deviance", "converged", "pnls_converged"):
            np.testing.assert_array_equal(getattr(updated, field), getattr(direct, field))


class TestNlmerVcov:
    def test_vcov_shape(self) -> None:
        result = fit_nlme()

        vcov = result.vcov()
        n_params = len(result.phi)
        assert vcov.shape == (n_params, n_params)

    def test_vcov_symmetric(self) -> None:
        result = fit_nlme()

        vcov = result.vcov()
        assert np.allclose(vcov, vcov.T)

    def test_vcov_positive_diagonal(self) -> None:
        result = fit_nlme()

        vcov = result.vcov()
        assert np.all(np.diag(vcov) >= 0)


class TestNlmerConfint:
    def test_confint_wald(self) -> None:
        result = fit_nlme()

        ci = result.confint(method="Wald", level=0.95)
        assert "Asym" in ci
        assert "R0" in ci
        assert "lrc" in ci

        for name, (lower, upper) in ci.items():
            if not np.isnan(lower) and not np.isnan(upper) and lower != upper:
                assert lower < upper
                assert lower < result.phi[result.model.param_names.index(name)]
                assert upper > result.phi[result.model.param_names.index(name)]

    def test_confint_bootstrap(self) -> None:
        result = fit_nlme()

        ci = result.confint(method="boot", n_boot=20, seed=42)
        assert "Asym" in ci
        for _name, (lower, upper) in ci.items():
            assert np.isfinite(lower) and np.isfinite(upper)
            assert lower < upper

    def test_confint_specific_params(self) -> None:
        result = fit_nlme()

        ci = result.confint(parm=["Asym"], method="Wald")
        assert "Asym" in ci
        assert "R0" not in ci

    def test_confint_invalid_method_raises(self) -> None:
        result = fit_nlme()

        with pytest.raises(ValueError, match="Unknown method"):
            result.confint(method="invalid")


class TestNlmerInfluence:
    def test_hatvalues(self) -> None:
        result = fit_nlme()

        h = result.hatvalues()
        assert len(h) == len(NLME_DATA)
        assert np.all(h >= 0)
        assert np.all(h < 1)

    def test_cooks_distance(self) -> None:
        result = fit_nlme()

        cooks_d = result.cooks_distance()
        assert len(cooks_d) == len(NLME_DATA)
        assert np.all(cooks_d >= 0)

    def test_influence_dict(self) -> None:
        result = fit_nlme()

        infl = result.influence()
        assert "hat" in infl
        assert "cooks_d" in infl
        assert "std_resid" in infl
        assert len(infl["hat"]) == len(NLME_DATA)


class TestNlmerGetME:
    def test_getME_phi(self) -> None:
        result = fit_nlme()

        phi = result.getME("phi")
        assert np.allclose(phi, result.phi)

    def test_getME_theta(self) -> None:
        result = fit_nlme()

        theta = result.getME("theta")
        assert np.allclose(theta, result.theta)

    def test_getME_sigma(self) -> None:
        result = fit_nlme()

        sigma = result.getME("sigma")
        assert sigma == result.sigma

    def test_getME_b(self) -> None:
        result = fit_nlme()

        b = result.getME("b")
        assert np.allclose(b, result.b)

    def test_getME_n_obs(self) -> None:
        result = fit_nlme()

        n = result.getME("n_obs")
        assert n == len(NLME_DATA)

    def test_getME_n_groups(self) -> None:
        result = fit_nlme()

        n_groups = result.getME("n_groups")
        assert n_groups == 8

    def test_getME_invalid_raises(self) -> None:
        result = fit_nlme()

        with pytest.raises(ValueError, match="Unknown component"):
            result.getME("invalid_name")


class TestNlmerIsSingular:
    def test_is_singular_normal_fit(self) -> None:
        result = fit_nlme()

        assert isinstance(result.isSingular(), bool)
        assert result.is_singular() == result.isSingular()

    def test_is_singular_with_tolerance(self) -> None:
        result = fit_nlme()

        result_strict = result.isSingular(tol=1e-2)
        result_loose = result.isSingular(tol=1e-10)
        assert isinstance(result_strict, bool)
        assert isinstance(result_loose, bool)


class TestNlmerAccessors:
    def test_root_accessor_functions(self) -> None:
        result = fit_nlme()

        assert fixef(result) == result.fixef()
        assert set(ranef(result)) == {"subject"}
        assert set(coef(result)) == {"subject"}
        assert np.allclose(getME(result, "phi"), result.phi)

    def test_nobs(self) -> None:
        result = fit_nlme()

        assert result.nobs() == len(NLME_DATA)

    def test_ngrps(self) -> None:
        result = fit_nlme()

        ngrps = result.ngrps()
        assert "subject" in ngrps
        assert ngrps["subject"] == 8

    def test_model_frame(self) -> None:
        result = fit_nlme()

        mf = result.model_frame()
        assert isinstance(mf, pd.DataFrame)
        assert len(mf) == len(NLME_DATA)


class TestNlmerWeightsOffset:
    def test_weights_default(self) -> None:
        result = fit_nlme()

        w = result.weights()
        assert len(w) == len(NLME_DATA)
        assert np.allclose(w, 1.0)

    def test_weights_specified(self) -> None:
        weights = np.random.uniform(0.5, 1.5, len(NLME_DATA))
        result = fit_nlme(weights=weights)

        w = result.weights()
        assert np.allclose(w, weights)

        expected_pearson = np.sqrt(weights) * result.residuals("response") / result.sigma
        assert np.allclose(result.residuals("pearson"), expected_pearson)

    def test_offset_default(self) -> None:
        result = fit_nlme()

        off = result.offset()
        assert len(off) == len(NLME_DATA)
        assert np.allclose(off, 0.0)

    def test_offset_specified(self) -> None:
        offset = np.random.randn(len(NLME_DATA)) * 0.1
        result = fit_nlme(offset=offset)

        off = result.offset()
        assert np.allclose(off, offset)

    def test_offset_applied_consistently(self) -> None:
        data = create_offset_nlme_data()
        offset = np.linspace(-2.0, 2.0, len(data))
        adjusted_data = data.assign(y=data["y"].to_numpy() - offset)
        fit_kwargs = {
            "x_var": "time",
            "y_var": "y",
            "group_var": "subject",
            "random_params": ["Asym"],
            "pnls_maxiter": 2000,
        }

        with_offset = nlmer(nlme.SSasymp(), data, offset=offset, **fit_kwargs)
        without_offset = nlmer(nlme.SSasymp(), adjusted_data, **fit_kwargs)

        assert with_offset.converged
        assert without_offset.converged
        np.testing.assert_allclose(with_offset.phi, without_offset.phi)
        np.testing.assert_allclose(with_offset.theta, without_offset.theta)
        np.testing.assert_allclose(with_offset.b, without_offset.b)
        np.testing.assert_allclose(with_offset.y, data["y"].to_numpy())
        np.testing.assert_allclose(with_offset.getME("y"), data["y"].to_numpy())
        np.testing.assert_allclose(with_offset.fitted(), without_offset.fitted() + offset)
        np.testing.assert_allclose(with_offset.residuals(), without_offset.residuals())
        np.testing.assert_allclose(with_offset.vcov(), without_offset.vcov())
        np.testing.assert_allclose(with_offset.hatvalues(), without_offset.hatvalues())

        simulated = with_offset.simulate(seed=123, use_re=False)
        adjusted_simulated = without_offset.simulate(seed=123, use_re=False)
        np.testing.assert_allclose(simulated, adjusted_simulated + offset)

        refitted = with_offset.refit(simulated)
        adjusted_refitted = without_offset.refit(adjusted_simulated)
        assert refitted.converged
        assert adjusted_refitted.converged
        np.testing.assert_allclose(refitted.phi, adjusted_refitted.phi, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(refitted.theta, adjusted_refitted.theta, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(refitted.b, adjusted_refitted.b, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(refitted.deviance, adjusted_refitted.deviance)
        np.testing.assert_allclose(refitted.y, simulated)
        np.testing.assert_allclose(refitted.model_frame()["y"], simulated)

    def test_weights_wrong_length_raises(self) -> None:
        model = nlme.SSasymp()
        with pytest.raises(ValueError, match="weights has length"):
            nlmer(
                model,
                NLME_DATA,
                x_var="time",
                y_var="y",
                group_var="subject",
                weights=np.array([1, 2, 3]),
            )

    @pytest.mark.parametrize(
        ("weights", "message"),
        [
            (np.zeros(len(NLME_DATA)), "strictly positive"),
            (np.full(len(NLME_DATA), np.inf), "finite values"),
            (np.ones((len(NLME_DATA), 1)), "one-dimensional"),
        ],
    )
    def test_invalid_weights_raise(self, weights, message) -> None:
        with pytest.raises(ValueError, match=message):
            nlmer(
                nlme.SSasymp(),
                NLME_DATA,
                x_var="time",
                y_var="y",
                group_var="subject",
                weights=weights,
            )

    def test_offset_wrong_length_raises(self) -> None:
        model = nlme.SSasymp()
        with pytest.raises(ValueError, match="offset has length"):
            nlmer(
                model,
                NLME_DATA,
                x_var="time",
                y_var="y",
                group_var="subject",
                offset=np.array([1, 2, 3]),
            )


class TestBootstrapNlmer:
    def test_bootstrap_nlmer_basic(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(
            model,
            create_offset_nlme_data(),
            x_var="time",
            y_var="y",
            group_var="subject",
            random_params=["Asym"],
            pnls_maxiter=2000,
        )
        assert result.converged and result.pnls_converged

        boot = bootstrap_nlmer(result, n_boot=10, seed=42)
        assert boot.n_boot == 10
        assert boot.phi_samples.shape == (10, len(result.phi))
        assert boot.theta_samples.shape == (10, len(result.theta))

    def test_bootstrap_nlmer_ci(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(
            model,
            create_offset_nlme_data(),
            x_var="time",
            y_var="y",
            group_var="subject",
            random_params=["Asym"],
            pnls_maxiter=2000,
        )
        assert result.converged and result.pnls_converged

        boot = bootstrap_nlmer(result, n_boot=20, seed=42)
        ci = boot.ci(level=0.95)

        for name in result.model.param_names:
            assert name in ci
            lower, upper = ci[name]
            assert lower < upper

    def test_bootstrap_nlmer_se(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(
            model,
            create_offset_nlme_data(),
            x_var="time",
            y_var="y",
            group_var="subject",
            random_params=["Asym"],
            pnls_maxiter=2000,
        )
        assert result.converged and result.pnls_converged

        boot = bootstrap_nlmer(result, n_boot=20, seed=42)
        se = boot.se()

        for name in result.model.param_names:
            assert name in se
            assert se[name] > 0

    def test_bootstrap_nlmer_summary(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(
            model,
            create_offset_nlme_data(),
            x_var="time",
            y_var="y",
            group_var="subject",
            random_params=["Asym"],
            pnls_maxiter=2000,
        )
        assert result.converged and result.pnls_converged

        boot = bootstrap_nlmer(result, n_boot=10, seed=42)
        summary = boot.summary()

        assert "Parametric bootstrap" in summary
        assert "Asym" in summary
