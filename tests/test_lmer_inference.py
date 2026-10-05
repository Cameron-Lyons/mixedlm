"""Confidence intervals, bootstrap, anova, simulation, prior weights and offsets."""

import numpy as np
import pandas as pd
import pytest
from mixedlm import (
    anova,
    families,
    glmer,
    lmer,
)
from numpy.testing import assert_allclose
from scipy import stats

from tests._datasets import CBPP, SLEEPSTUDY, grouped_data

Z_975 = stats.norm.ppf(0.975)


@pytest.fixture(scope="module")
def sleepstudy_bootstrap(sleepstudy_lmm):
    from mixedlm.inference import bootstrap_lmer

    return bootstrap_lmer(sleepstudy_lmm, n_boot=30, seed=42)


@pytest.fixture(scope="module")
def nested_ml_fits():
    return [
        lmer(formula, SLEEPSTUDY, REML=False)
        for formula in (
            "Reaction ~ 1 + (1 | Subject)",
            "Reaction ~ Days + (1 | Subject)",
            "Reaction ~ Days + (Days | Subject)",
        )
    ]


class TestConfidenceIntervals:
    @pytest.mark.parametrize("model", ["sleepstudy_lmm", "cbpp_glmm"])
    def test_wald_intervals_use_normal_quantiles(self, request, model) -> None:
        result = request.getfixturevalue(model)
        se = np.sqrt(np.diag(result.vcov()))

        ci = result.confint(method="Wald")

        assert list(ci) == result.matrices.fixed_names
        assert_allclose(
            [ci[name] for name in ci],
            np.column_stack((result.beta - Z_975 * se, result.beta + Z_975 * se)),
        )

    def test_profile_interval_matches_profile_lmer(self, sleepstudy_lmm) -> None:
        from mixedlm.inference import profile_lmer

        ci = sleepstudy_lmm.confint(parm="Days", method="profile")
        profile = profile_lmer(sleepstudy_lmm, which="Days", n_points=10)["Days"]

        assert ci["Days"] == pytest.approx((profile.ci_lower, profile.ci_upper))
        assert len(profile.values) == len(profile.zeta) == 10
        assert profile.ci_lower < sleepstudy_lmm.beta[1] < profile.ci_upper
        # The profile is asymmetric but close to Wald for a well-determined slope.
        wald = sleepstudy_lmm.confint(parm="Days", method="Wald")["Days"]
        assert ci["Days"] == pytest.approx(wald, abs=0.05)

    def test_glmm_profile_interval_brackets_the_estimate(self, cbpp_glmm) -> None:
        ci = cbpp_glmm.confint(parm="(Intercept)", method="profile")

        assert ci["(Intercept)"][0] < cbpp_glmm.beta[0] < ci["(Intercept)"][1]

    def test_bootstrap_interval_is_the_percentile_interval(self, sleepstudy_lmm) -> None:
        from mixedlm.inference import bootstrap_lmer

        ci = sleepstudy_lmm.confint(parm="Days", method="boot", n_boot=30, seed=42)
        boot = bootstrap_lmer(sleepstudy_lmm, n_boot=30, seed=42)

        assert ci["Days"] == pytest.approx(boot.ci(method="percentile")["Days"])


class TestBootstrap:
    def test_samples_and_standard_errors(self, sleepstudy_lmm, sleepstudy_bootstrap) -> None:
        boot = sleepstudy_bootstrap

        assert boot.n_boot == 30
        assert boot.n_failed == 0
        assert boot.beta_samples.shape == (30, 2)
        se = boot.se()
        assert list(se) == ["(Intercept)", "Days"]
        assert se["Days"] == pytest.approx(np.std(boot.beta_samples[:, 1], ddof=1))
        # Thirty draws estimate the Wald SE to within roughly a third.
        assert se["Days"] == pytest.approx(np.sqrt(sleepstudy_lmm.vcov()[1, 1]), rel=0.35)

    def test_interval_methods(self, sleepstudy_lmm, sleepstudy_bootstrap) -> None:
        samples = sleepstudy_bootstrap.beta_samples[:, 1]
        estimate = sleepstudy_lmm.beta[1]
        percentile = np.percentile(samples, [2.5, 97.5])
        # The normal interval is bias corrected, as in R's boot::norm.ci.
        centre = 2 * estimate - samples.mean()

        assert_allclose(sleepstudy_bootstrap.ci(method="percentile")["Days"], percentile)
        assert_allclose(
            sleepstudy_bootstrap.ci(method="basic")["Days"], 2 * estimate - percentile[::-1]
        )
        assert_allclose(
            sleepstudy_bootstrap.ci(method="normal")["Days"],
            centre + np.array([-1, 1]) * Z_975 * np.std(samples, ddof=1),
        )

    def test_summary(self, sleepstudy_bootstrap) -> None:
        summary = sleepstudy_bootstrap.summary()

        assert "Parametric bootstrap" in summary
        assert "30 samples" in summary

    def test_glmm_bootstrap_has_no_residual_scale(self, cbpp_glmm) -> None:
        from mixedlm.inference import bootstrap_glmer

        boot = bootstrap_glmer(cbpp_glmm, n_boot=5, seed=42)

        assert boot.n_boot == 5
        assert boot.beta_samples.shape == (5, 4)
        assert boot.sigma_samples is None


class TestAnova:
    def test_likelihood_ratio_test(self, nested_ml_fits) -> None:
        reduced, full = nested_ml_fits[:2]

        result = anova(reduced, full)

        chi_sq = 2 * (full.logLik().value - reduced.logLik().value)
        assert len(result.models) == 2
        assert result.chi_sq[0] is None
        assert result.chi_sq[1] == pytest.approx(chi_sq)
        assert result.chi_df[1] == 1
        assert result.p_value[1] == pytest.approx(stats.chi2.sf(chi_sq, 1))
        assert result.aic == pytest.approx([reduced.AIC(), full.AIC()])
        assert result.bic == pytest.approx([reduced.BIC(), full.BIC()])
        output = str(result)
        for text in ("AIC", "BIC", "logLik", "Chisq"):
            assert text in output

    def test_three_nested_models(self, nested_ml_fits) -> None:
        result = anova(*nested_ml_fits)

        assert len(result.models) == 3
        assert result.chi_df[1:] == [1, 2]
        assert result.aic == pytest.approx([model.AIC() for model in nested_ml_fits])

    def test_reml_models_are_refitted_with_ml(self, sleepstudy_lmm, nested_ml_fits) -> None:
        intercept_only = lmer("Reaction ~ 1 + (1 | Subject)", SLEEPSTUDY)

        result = anova(intercept_only, sleepstudy_lmm)

        # Already-ML fits take the no-refit path without a REML warning.
        expected = anova(*nested_ml_fits[:2], refit=False)
        assert intercept_only.REML is True
        assert sleepstudy_lmm.REML is True
        assert result.loglik == pytest.approx(expected.loglik)
        assert result.aic == pytest.approx(expected.aic)
        assert result.bic == pytest.approx(expected.bic)
        assert result.chi_sq[1] == pytest.approx(expected.chi_sq[1])
        assert result.p_value[1] == pytest.approx(expected.p_value[1])

    def test_can_skip_ml_refit(self, sleepstudy_lmm) -> None:
        intercept_only = lmer("Reaction ~ 1 + (1 | Subject)", SLEEPSTUDY)

        with pytest.warns(UserWarning, match="refit=False"):
            result = anova(intercept_only, sleepstudy_lmm, refit=False)

        assert result.loglik == pytest.approx(
            [intercept_only.logLik().value, sleepstudy_lmm.logLik().value]
        )

    def test_glmm_likelihood_ratio_test(self, cbpp_glmm) -> None:
        reduced = glmer("incidence / size ~ 1 + (1 | herd)", CBPP, family=families.Binomial())

        result = anova(reduced, cbpp_glmm)

        chi_sq = 2 * (cbpp_glmm.logLik().value - reduced.logLik().value)
        assert result.chi_df[1] == 3
        assert result.chi_sq[1] == pytest.approx(chi_sq)
        assert result.p_value[1] == pytest.approx(stats.chi2.sf(chi_sq, 3))


class TestSimulate:
    def test_draws_are_seeded_and_shaped(self, sleepstudy_lmm) -> None:
        single = sleepstudy_lmm.simulate(nsim=1, seed=42)
        multiple = sleepstudy_lmm.simulate(nsim=10, seed=42)

        assert single.shape == (180,)
        assert multiple.shape == (180, 10)
        assert np.isfinite(multiple).all()
        assert_allclose(sleepstudy_lmm.simulate(nsim=1, seed=42), single)
        assert not np.allclose(sleepstudy_lmm.simulate(nsim=1, seed=42, use_re=False), single)

    def test_multiple_draws_preserve_seeded_random_effects(self, sleepstudy_slopes_lmm) -> None:
        from mixedlm._rust import simulate_re_batch

        result = sleepstudy_slopes_lmm
        nsim = 5
        seed = 42
        structures = result.matrices.random_structures
        u_batch = simulate_re_batch(
            result.theta,
            result.sigma,
            [struct.n_levels for struct in structures],
            [struct.n_terms for struct in structures],
            [struct.correlated for struct in structures],
            nsim,
            seed,
        )

        np.random.seed(seed)
        expected = np.column_stack(
            [
                result.matrices.X @ result.beta
                + result.matrices.Z @ u_batch[i]
                + np.random.randn(result.matrices.n_obs) * result.sigma
                for i in range(nsim)
            ]
        )

        simulated = result.simulate(nsim=nsim, seed=seed)

        assert_allclose(simulated, expected)

    def test_multiple_draws_without_random_effects_preserve_seeded_draws(
        self, sleepstudy_lmm
    ) -> None:
        nsim = 5
        seed = 42

        np.random.seed(seed)
        expected = np.column_stack(
            [sleepstudy_lmm._simulate_once(use_re=False) for _ in range(nsim)]
        )

        simulated = sleepstudy_lmm.simulate(nsim=nsim, seed=seed, use_re=False)

        assert_allclose(simulated, expected)

    @pytest.mark.parametrize("model", ["sleepstudy_lmm", "grouped_glmm"])
    def test_rejects_nonpositive_nsim(self, request, model) -> None:
        with pytest.raises(ValueError, match="nsim must be a positive integer"):
            request.getfixturevalue(model).simulate(nsim=0)

    def test_bernoulli_draws_use_seeded_random_effects(self, grouped_glmm) -> None:
        from mixedlm._rust import simulate_re_batch

        result = grouped_glmm
        nsim = 5
        seed = 42
        structures = result.matrices.random_structures
        u_batch = simulate_re_batch(
            result.theta,
            1.0,
            [structure.n_levels for structure in structures],
            [structure.n_terms for structure in structures],
            [structure.correlated for structure in structures],
            nsim,
            seed,
        )
        eta = (result.matrices.X @ result.beta + result.matrices.offset)[:, None] + np.asarray(
            result.matrices.Z @ u_batch.T
        )
        mu = np.clip(result.family.link.inverse(eta), 1e-6, 1 - 1e-6)
        np.random.seed(seed)
        expected = np.random.binomial(1, mu).astype(np.float64)

        y_sim = result.simulate(nsim=nsim, seed=seed)

        assert y_sim.shape == (200, 5)
        assert_allclose(y_sim, expected)
        assert_allclose(result.simulate(nsim=nsim, seed=seed), expected)

    def test_glmm_simulation_uses_family_subclasses(self, monkeypatch) -> None:
        # Reassigns the family, so it needs a private fit.
        result = glmer("y ~ x + (1 | group)", grouped_data("binomial"), family=families.Binomial())
        mu = np.full((result.matrices.n_obs, 3), 0.75)
        monkeypatch.setattr(np.random, "gamma", lambda shape, scale: np.asarray(scale))
        monkeypatch.setattr(np.random, "wald", lambda mean, scale: np.asarray(mean))

        result.family = families.GammaInverse()
        gamma_draws = result._simulate_response(mu)
        result.family = families.InverseGaussianCanonical()
        inverse_gaussian_draws = result._simulate_response(mu)

        assert_allclose(gamma_draws, mu)
        assert_allclose(inverse_gaussian_draws, mu)

    def test_poisson_draws_are_counts_around_the_fitted_means(self) -> None:
        result = glmer("y ~ x + (1 | group)", grouped_data("poisson"), family=families.Poisson())

        y_sim = result.simulate(nsim=200, seed=42, use_re=False)

        assert y_sim.shape == (200, 200)
        assert np.all(y_sim >= 0)
        assert np.all(y_sim == np.round(y_sim))
        marginal = np.exp(result.matrices.X @ result.beta)
        # Each row averages 200 Poisson draws: a 5-SE band holds every row.
        assert np.all(np.abs(y_sim.mean(axis=1) - marginal) < 5 * np.sqrt(marginal / 200))


class TestWeightsOffset:
    def test_rescaled_weights_keep_the_random_effect_scale(self) -> None:
        data = grouped_data()
        weights = np.random.default_rng(7).uniform(0.1, 2.0, len(data))

        unweighted = lmer("y ~ x + (1 | group)", data)
        weighted = lmer("y ~ x + (1 | group)", data, weights=weights)
        doubled = lmer("y ~ x + (1 | group)", data, weights=2 * weights)

        assert weighted.converged
        assert not np.allclose(weighted.beta, unweighted.beta)
        # Weights scale only the residual variance: sigma grows by sqrt(2) while
        # the absolute random-effect SD, theta * sigma, is unchanged.
        assert_allclose(doubled.beta, weighted.beta, atol=1e-6)
        assert doubled.sigma == pytest.approx(weighted.sigma * np.sqrt(2), rel=1e-5)
        assert doubled.theta[0] * doubled.sigma == pytest.approx(
            weighted.theta[0] * weighted.sigma, rel=1e-4
        )

    def test_offset_is_equivalent_to_shifting_the_response(self) -> None:
        data = grouped_data()
        offset = np.random.default_rng(7).normal(0.0, 0.5, len(data))
        weights = np.random.default_rng(8).uniform(0.1, 2.0, len(data))

        with_offset = lmer("y ~ x + (1 | group)", data, weights=weights, offset=offset)
        shifted = lmer("y ~ x + (1 | group)", data.assign(y=data["y"] - offset), weights=weights)

        assert_allclose(with_offset.beta, shifted.beta, atol=1e-6)
        assert_allclose(with_offset.theta, shifted.theta, atol=1e-5)
        assert_allclose(with_offset.fitted(), shifted.fitted() + offset, atol=1e-5)

    def test_lmer_simulate_preserves_offset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        n_groups = 8
        n_per_group = 10
        n = n_groups * n_per_group
        rng = np.random.default_rng(42)
        group = np.repeat(np.arange(n_groups), n_per_group)
        x = rng.normal(size=n)
        offset = np.linspace(-0.4, 0.6, n)
        y = 1.5 + 0.3 * x + offset + rng.normal(0.0, 0.2, n)
        data = pd.DataFrame({"y": y, "x": x, "group": group.astype(str)})
        # The simulated means have no group effect.
        with pytest.warns(UserWarning, match="Model is singular"):
            result = lmer("y ~ x + (1 | group)", data, offset=offset)

        monkeypatch.setattr(result, "sigma", 0.0)
        simulated = result.simulate(nsim=3, use_re=False)

        expected = result.matrices.X @ result.beta + offset
        assert_allclose(simulated, np.broadcast_to(expected[:, None], (n, 3)))

    def test_integer_binomial_weights_replicate_observations(self) -> None:
        data = grouped_data("binomial")

        weighted = glmer(
            "y ~ x + (1 | group)", data, family=families.Binomial(), weights=np.full(200, 2.0)
        )
        replicated = glmer(
            "y ~ x + (1 | group)", pd.concat([data, data]), family=families.Binomial()
        )

        assert weighted.converged
        assert_allclose(weighted.beta, replicated.beta, atol=1e-4)
        assert_allclose(weighted.theta, replicated.theta, atol=1e-4)

    def test_glmer_offset_enters_the_linear_predictor(self) -> None:
        data = grouped_data("poisson")
        log_exposure = np.random.default_rng(7).normal(0.0, 0.5, len(data))

        result = glmer("y ~ x + (1 | group)", data, family=families.Poisson(), offset=log_exposure)

        eta = result.getME("X") @ result.beta + result.getME("Z") @ result.getME("b")
        assert result.converged
        assert_allclose(result.fitted(), np.exp(eta + log_exposure))

    def test_glmer_simulate_preserves_offset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        n_groups = 8
        n_per_group = 10
        n = n_groups * n_per_group
        rng = np.random.default_rng(42)
        group = np.repeat(np.arange(n_groups), n_per_group)
        x = rng.normal(size=n)
        offset = np.linspace(-0.3, 0.5, n)
        mu = np.exp(-0.7 + 0.2 * x + offset)
        y = rng.poisson(mu)
        data = pd.DataFrame({"y": y, "x": x, "group": group.astype(str)})
        # The simulated means have no group effect.
        with pytest.warns(UserWarning, match="Model is singular"):
            result = glmer(
                "y ~ x + (1 | group)",
                data,
                family=families.Poisson(),
                offset=offset,
            )

        monkeypatch.setattr(result.family, "simulate", lambda mu, rng=None: np.asarray(mu))
        simulated = result.simulate(nsim=3, use_re=False)

        eta = result.matrices.X @ result.beta + offset
        expected = result.family.link.inverse(eta)
        assert_allclose(simulated, np.broadcast_to(expected[:, None], (n, 3)))

    def test_glmer_simulation_and_bootstrap_preserve_offset(self, monkeypatch) -> None:
        from mixedlm.inference.bootstrap import _prepare_glmer_worker_data

        rng = np.random.default_rng(17)
        n_groups = 6
        n_per_group = 8
        n = n_groups * n_per_group
        group = np.repeat(np.arange(n_groups), n_per_group)
        x = rng.normal(size=n)
        offset = np.linspace(-0.7, 0.7, n)
        y = rng.poisson(np.exp(1.0 + 0.2 * x + offset))
        data = pd.DataFrame({"y": y, "x": x, "group": group.astype(str)})
        # The simulated means have no group effect.
        with pytest.warns(UserWarning, match="Model is singular"):
            result = glmer(
                "y ~ x + (1 | group)",
                data,
                family=families.Poisson(),
                offset=offset,
            )
        captured: dict[str, np.ndarray] = {}

        def record_mu(mu, rng=None):
            captured["mu"] = mu.copy()
            return np.zeros_like(mu)

        monkeypatch.setattr(result.family, "simulate", record_mu)

        result.simulate(use_re=False)

        expected = result.family.link.inverse(result.matrices.X @ result.beta + offset)
        np.testing.assert_allclose(captured["mu"], expected)
        worker_data = _prepare_glmer_worker_data(result)
        worker_matrices = worker_data["matrices"]
        np.testing.assert_allclose(worker_matrices.offset, offset)
        np.testing.assert_allclose(worker_matrices.weights, result.matrices.weights)
