from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mixedlm import nlme, nlmer
from mixedlm.inference.bootstrap import bootstrap_nlmer


def create_nlme_data(n_groups: int = 8, n_per_group: int = 10, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    data_rows = []
    for subj in range(n_groups):
        asym = 200 + np.random.randn() * 20
        r0 = 180 + np.random.randn() * 10
        lrc = -3 + np.random.randn() * 0.2
        for t in np.linspace(0, 10, n_per_group):
            y = asym - (asym - r0) * np.exp(np.exp(lrc) * t) + np.random.randn() * 5
            data_rows.append({"subject": f"S{subj + 1}", "time": t, "y": y})
    return pd.DataFrame(data_rows)


NLME_DATA = create_nlme_data()


class TestNlmerSimulate:
    def test_simulate_single(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(model, NLME_DATA, x_var="time", y_var="y", group_var="subject")

        sim = result.simulate(nsim=1, seed=123)
        assert sim.shape == (len(NLME_DATA),)
        assert not np.allclose(sim, result.y)

    def test_simulate_multiple(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(model, NLME_DATA, x_var="time", y_var="y", group_var="subject")

        sim = result.simulate(nsim=5, seed=123)
        assert sim.shape == (len(NLME_DATA), 5)

    def test_simulate_reproducible(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(model, NLME_DATA, x_var="time", y_var="y", group_var="subject")

        sim1 = result.simulate(nsim=1, seed=42)
        sim2 = result.simulate(nsim=1, seed=42)
        assert np.allclose(sim1, sim2)

    def test_simulate_no_random_effects(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(model, NLME_DATA, x_var="time", y_var="y", group_var="subject")

        sim_with_re = result.simulate(nsim=1, seed=123, use_re=True)
        sim_no_re = result.simulate(nsim=1, seed=123, use_re=False)
        assert not np.allclose(sim_with_re, sim_no_re)

    def test_simulate_re_form_na(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(model, NLME_DATA, x_var="time", y_var="y", group_var="subject")

        sim = result.simulate(nsim=1, seed=123, re_form="NA")
        assert sim.shape == (len(NLME_DATA),)


class TestNlmerRefit:
    def test_refit_same_response(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(model, NLME_DATA, x_var="time", y_var="y", group_var="subject")

        refit_result = result.refit()
        assert len(refit_result.phi) == len(result.phi)
        assert refit_result.deviance is not None
        if refit_result.converged and not np.any(np.isnan(refit_result.phi)):
            assert np.allclose(result.phi, refit_result.phi, atol=1.0)

    def test_refit_new_response(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(model, NLME_DATA, x_var="time", y_var="y", group_var="subject")

        new_y = result.simulate(nsim=1, seed=456)
        refit_result = result.refit(new_y)

        assert len(refit_result.phi) == len(result.phi)
        assert np.allclose(refit_result.y, new_y)

    def test_refit_wrong_length_raises(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(model, NLME_DATA, x_var="time", y_var="y", group_var="subject")

        with pytest.raises(ValueError, match="newresp has length"):
            result.refit(np.array([1, 2, 3]))


class TestNlmerUpdate:
    def test_update_same_data(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(model, NLME_DATA, x_var="time", y_var="y", group_var="subject")

        updated = result.update()
        assert updated.converged
        assert np.allclose(result.phi, updated.phi, atol=0.1)

    def test_update_with_start(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(model, NLME_DATA, x_var="time", y_var="y", group_var="subject")

        start = {"Asym": 200.0, "R0": 180.0, "lrc": -3.0}
        updated = result.update(start=start)
        assert updated.converged


class TestNlmerVcov:
    def test_vcov_shape(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(model, NLME_DATA, x_var="time", y_var="y", group_var="subject")

        vcov = result.vcov()
        n_params = len(result.phi)
        assert vcov.shape == (n_params, n_params)

    def test_vcov_symmetric(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(model, NLME_DATA, x_var="time", y_var="y", group_var="subject")

        vcov = result.vcov()
        assert np.allclose(vcov, vcov.T)

    def test_vcov_positive_diagonal(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(model, NLME_DATA, x_var="time", y_var="y", group_var="subject")

        vcov = result.vcov()
        assert np.all(np.diag(vcov) >= 0)


class TestNlmerConfint:
    def test_confint_wald(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(model, NLME_DATA, x_var="time", y_var="y", group_var="subject")

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
        model = nlme.SSasymp()
        result = nlmer(model, NLME_DATA, x_var="time", y_var="y", group_var="subject")

        ci = result.confint(method="boot", n_boot=20, seed=42)
        assert "Asym" in ci
        for name, (lower, upper) in ci.items():
            if not np.isnan(lower) and not np.isnan(upper):
                assert lower < upper

    def test_confint_specific_params(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(model, NLME_DATA, x_var="time", y_var="y", group_var="subject")

        ci = result.confint(parm=["Asym"], method="Wald")
        assert "Asym" in ci
        assert "R0" not in ci

    def test_confint_invalid_method_raises(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(model, NLME_DATA, x_var="time", y_var="y", group_var="subject")

        with pytest.raises(ValueError, match="Unknown method"):
            result.confint(method="invalid")


class TestNlmerInfluence:
    def test_hatvalues(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(model, NLME_DATA, x_var="time", y_var="y", group_var="subject")

        h = result.hatvalues()
        assert len(h) == len(NLME_DATA)
        assert np.all(h >= 0)
        assert np.all(h < 1)

    def test_cooks_distance(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(model, NLME_DATA, x_var="time", y_var="y", group_var="subject")

        cooks_d = result.cooks_distance()
        assert len(cooks_d) == len(NLME_DATA)
        assert np.all(cooks_d >= 0)

    def test_influence_dict(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(model, NLME_DATA, x_var="time", y_var="y", group_var="subject")

        infl = result.influence()
        assert "hat" in infl
        assert "cooks_d" in infl
        assert "std_resid" in infl
        assert len(infl["hat"]) == len(NLME_DATA)


class TestNlmerGetME:
    def test_getME_phi(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(model, NLME_DATA, x_var="time", y_var="y", group_var="subject")

        phi = result.getME("phi")
        assert np.allclose(phi, result.phi)

    def test_getME_theta(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(model, NLME_DATA, x_var="time", y_var="y", group_var="subject")

        theta = result.getME("theta")
        assert np.allclose(theta, result.theta)

    def test_getME_sigma(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(model, NLME_DATA, x_var="time", y_var="y", group_var="subject")

        sigma = result.getME("sigma")
        assert sigma == result.sigma

    def test_getME_b(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(model, NLME_DATA, x_var="time", y_var="y", group_var="subject")

        b = result.getME("b")
        assert np.allclose(b, result.b)

    def test_getME_n_obs(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(model, NLME_DATA, x_var="time", y_var="y", group_var="subject")

        n = result.getME("n_obs")
        assert n == len(NLME_DATA)

    def test_getME_n_groups(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(model, NLME_DATA, x_var="time", y_var="y", group_var="subject")

        n_groups = result.getME("n_groups")
        assert n_groups == 8

    def test_getME_invalid_raises(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(model, NLME_DATA, x_var="time", y_var="y", group_var="subject")

        with pytest.raises(ValueError, match="Unknown component"):
            result.getME("invalid_name")


class TestNlmerIsSingular:
    def test_is_singular_normal_fit(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(model, NLME_DATA, x_var="time", y_var="y", group_var="subject")

        assert isinstance(result.isSingular(), bool)

    def test_is_singular_with_tolerance(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(model, NLME_DATA, x_var="time", y_var="y", group_var="subject")

        result_strict = result.isSingular(tol=1e-2)
        result_loose = result.isSingular(tol=1e-10)
        assert isinstance(result_strict, bool)
        assert isinstance(result_loose, bool)


class TestNlmerAccessors:
    def test_nobs(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(model, NLME_DATA, x_var="time", y_var="y", group_var="subject")

        assert result.nobs() == len(NLME_DATA)

    def test_ngrps(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(model, NLME_DATA, x_var="time", y_var="y", group_var="subject")

        ngrps = result.ngrps()
        assert "subject" in ngrps
        assert ngrps["subject"] == 8

    def test_model_frame(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(model, NLME_DATA, x_var="time", y_var="y", group_var="subject")

        mf = result.model_frame()
        assert isinstance(mf, pd.DataFrame)
        assert len(mf) == len(NLME_DATA)


class TestNlmerWeightsOffset:
    def test_weights_default(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(model, NLME_DATA, x_var="time", y_var="y", group_var="subject")

        w = result.weights()
        assert len(w) == len(NLME_DATA)
        assert np.allclose(w, 1.0)

    def test_weights_specified(self) -> None:
        model = nlme.SSasymp()
        weights = np.random.uniform(0.5, 1.5, len(NLME_DATA))
        result = nlmer(
            model, NLME_DATA, x_var="time", y_var="y", group_var="subject", weights=weights
        )

        w = result.weights()
        assert np.allclose(w, weights)

    def test_offset_default(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(model, NLME_DATA, x_var="time", y_var="y", group_var="subject")

        off = result.offset()
        assert len(off) == len(NLME_DATA)
        assert np.allclose(off, 0.0)

    def test_offset_specified(self) -> None:
        model = nlme.SSasymp()
        offset = np.random.randn(len(NLME_DATA)) * 0.1
        result = nlmer(
            model, NLME_DATA, x_var="time", y_var="y", group_var="subject", offset=offset
        )

        off = result.offset()
        assert np.allclose(off, offset)

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
        result = nlmer(model, NLME_DATA, x_var="time", y_var="y", group_var="subject")

        boot = bootstrap_nlmer(result, n_boot=10, seed=42)
        assert boot.n_boot == 10
        assert boot.phi_samples.shape == (10, len(result.phi))
        assert boot.theta_samples.shape == (10, len(result.theta))

    def test_bootstrap_nlmer_ci(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(model, NLME_DATA, x_var="time", y_var="y", group_var="subject")

        boot = bootstrap_nlmer(result, n_boot=20, seed=42)
        ci = boot.ci(level=0.95)

        for name in result.model.param_names:
            assert name in ci
            lower, upper = ci[name]
            assert lower < upper

    def test_bootstrap_nlmer_se(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(model, NLME_DATA, x_var="time", y_var="y", group_var="subject")

        boot = bootstrap_nlmer(result, n_boot=20, seed=42)
        se = boot.se()

        for name in result.model.param_names:
            assert name in se
            assert se[name] > 0

    def test_bootstrap_nlmer_summary(self) -> None:
        model = nlme.SSasymp()
        result = nlmer(model, NLME_DATA, x_var="time", y_var="y", group_var="subject")

        boot = bootstrap_nlmer(result, n_boot=10, seed=42)
        summary = boot.summary()

        assert "Parametric bootstrap" in summary
        assert "Asym" in summary
