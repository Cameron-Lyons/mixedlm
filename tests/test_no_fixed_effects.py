from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from mixedlm import glmer, lmer, lmerControl, parse_formula
from mixedlm.estimation import laplace, optimizers
from mixedlm.estimation.reml import LMMOptimizer, _build_lambda
from mixedlm.families import Binomial, Poisson
from mixedlm.inference import ddf
from mixedlm.matrices import build_model_matrices
from mixedlm.models.checks import check_rankX
from mixedlm.models.glmer import GlmerResult
from mixedlm.models.lmer import LmerResult
from numpy.testing import assert_allclose, assert_array_equal
from scipy import linalg, optimize, special


@pytest.fixture
def balanced_data():
    effects = np.array([0.2, 0.7, 1.5, -0.5, -1.0, 2.0, 1.2, -0.3])
    noise = np.array([-0.45, -0.25, -0.1, 0.1, 0.25, 0.45])
    groups = np.repeat(np.arange(len(effects)), len(noise))
    x = np.tile(noise, len(effects))
    offset = np.linspace(-0.3, 0.4, len(groups))
    return pd.DataFrame({"y": effects[groups] + x + offset, "x": x, "g": groups, "off": offset})


@pytest.mark.parametrize(
    "text",
    [
        "y ~ 0 + (1 | g)",
        "y ~ -1 + (1 | g)",
        "y ~ x - x - 1 + (1 | g)",
        "y ~ 0 + x - x + (1 | g)",
        "y ~ 0",
        "y ~ 1 - 1",
    ],
)
def test_explicitly_empty_fixed_formula_builds_zero_columns(balanced_data, text):
    formula = parse_formula(text)
    matrices = build_model_matrices(formula, balanced_data)

    assert not formula.fixed.has_intercept
    assert parse_formula(str(formula)) == formula
    assert matrices.X.shape == (len(balanced_data), 0)
    assert matrices.X.dtype == np.float64
    assert matrices.fixed_names == []
    assert matrices.n_fixed == 0
    assert matrices.n_random == (8 if formula.random else 0)


@pytest.mark.parametrize("action", ["ignore", "warning", "stop", "message+drop.cols"])
def test_rank_check_accepts_empty_fixed_matrix(balanced_data, action, monkeypatch):
    matrices = build_model_matrices(parse_formula("y ~ 0 + (1 | g)"), balanced_data)

    def unexpected_rank(*args, **kwargs):
        raise AssertionError("an empty fixed design needs no singular value decomposition")

    monkeypatch.setattr(np.linalg, "matrix_rank", unexpected_rank)
    matrix, dropped = check_rankX(matrices, action)

    assert matrix is matrices.X
    assert dropped is None


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("reml", [False, True])
@pytest.mark.parametrize("random", [False, True])
def test_lmm_without_fixed_effects_matches_closed_form(balanced_data, native, reml, random):
    data = balanced_data
    weight = 2.5
    model = lmer(
        "y ~ 0 + (1 | g)" if random else "y ~ 0",
        data,
        REML=reml,
        weights=np.full(len(data), weight),
        offset="off",
        control=lmerControl(use_rust=native),
    )
    adjusted = np.asarray(data.y - data.off)
    n = len(data)
    if random:
        means = adjusted.reshape(8, 6).mean(axis=1)
        within = adjusted - np.repeat(means, 6)
        sigma2 = weight * np.dot(within, within) / (n - 8)
        tau2 = np.mean(means**2) - sigma2 / (6 * weight)
        expected_u = means * tau2 / (tau2 + sigma2 / (6 * weight))
        assert_allclose(model.theta, [np.sqrt(tau2 / sigma2)], rtol=2e-5)
        assert_allclose(model.u, expected_u, rtol=2e-5)
        covariance = sigma2 / weight * np.eye(n) + tau2 * (
            np.asarray(data.g)[:, None] == np.asarray(data.g)[None, :]
        )
    else:
        sigma2 = weight * np.mean(adjusted**2)
        covariance = sigma2 / weight * np.eye(n)
        assert model.theta.size == model.u.size == 0
        assert model.n_iter == 0
        assert model.function_evals == 1
    expected_deviance = (
        n * np.log(2 * np.pi)
        + np.linalg.slogdet(covariance)[1]
        + adjusted @ linalg.solve(covariance, adjusted, assume_a="pos")
    )

    assert model.converged
    assert_allclose(model.sigma**2, sigma2, rtol=2e-5)
    assert_allclose(model.deviance, expected_deviance, atol=1e-8)
    assert model.beta.shape == (0,)
    assert model.fixef() == {}
    assert model.vcov().shape == (0, 0)
    assert model.confint() == {}
    assert model.df_residual() == n
    assert model.npar() == (2 if random else 1)
    assert "convergence: yes" in model.summary()
    assert_allclose(model.predict(re_form="NA"), data.off)
    assert_allclose(model.predict(data, offset="off"), model.fitted())

    prediction = model.predict(re_form="NA", se_fit=True, interval="confidence")
    assert_array_equal(prediction.se_fit, np.zeros(n))
    assert_array_equal(prediction.lower, data.off)
    assert_array_equal(prediction.upper, data.off)
    blank = pd.DataFrame(index=range(3))
    assert_array_equal(model.predict(blank, re_form="NA"), np.zeros(3))
    assert_array_equal(model.predict(blank, re_form="NA", offset=0.4), np.full(3, 0.4))
    if random:
        newdata = pd.DataFrame({"g": [0, 99], "off": [0.7, -0.2]})
        conditional = model.predict(newdata, offset="off", allow_new_levels=True, se_fit=True)
        fitted_tau2 = (model.theta[0] * model.sigma) ** 2
        posterior_variance = fitted_tau2 / (1 + 6 * weight * model.theta[0] ** 2)
        assert_allclose(conditional.fit, [model.u[0] + 0.7, -0.2])
        assert_allclose(conditional.se_fit**2, [posterior_variance, fitted_tau2])

    assert model.hatvalues().shape == (n,)
    assert np.all(np.isfinite(model.hatvalues()))
    with np.errstate(divide="raise", invalid="raise"):
        assert np.all(np.isnan(model.cooks_distance()))
        assert np.all(np.isnan(model.influence()["cooks_d"]))
    refit = model.refit(np.asarray(data.y))
    assert refit.beta.shape == (0,)
    assert_allclose(refit.fitted(), model.fitted(), rtol=2e-5, atol=1e-6)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("reml", [False, True])
@pytest.mark.parametrize("slopes", [False, True])
def test_empty_fixed_likelihood_matches_weighted_marginal_covariance(
    balanced_data, native, reml, slopes
):
    formula = parse_formula("y ~ 0 + (1 + x | g)" if slopes else "y ~ 0 + (1 | g)")
    matrices = build_model_matrices(
        formula,
        balanced_data,
        weights=np.linspace(0.4, 2.0, len(balanced_data)),
        offset=np.asarray(balanced_data.off),
    )
    theta = np.array([0.8, 0.1, 0.6] if slopes else [0.8])
    factor = _build_lambda(theta, matrices.random_structures).toarray()
    design = matrices.Z @ factor
    covariance = np.diag(1 / matrices.weights) + design @ design.T
    adjusted = matrices.y - matrices.offset
    solved = linalg.solve(covariance, adjusted, assume_a="pos")
    sigma2 = adjusted @ solved / matrices.n_obs
    expected = matrices.n_obs * (1 + np.log(2 * np.pi * sigma2))
    expected += np.linalg.slogdet(covariance)[1]
    optimizer = LMMOptimizer(matrices, REML=reml, use_rust=native)

    assert_allclose(optimizer.objective(theta), expected, rtol=1e-12)
    beta, sigma, effects = optimizer._extract_estimates(theta)
    assert beta.shape == (0,)
    assert_allclose(sigma**2, sigma2, rtol=1e-12)
    assert_allclose(effects, factor @ design.T @ solved, rtol=1e-12, atol=1e-12)


def _count_data(kind):
    rng = np.random.default_rng(622)
    groups = np.repeat(np.arange(8), 8)
    offset = np.linspace(-0.2, 0.2, len(groups))
    eta = np.linspace(-1.0, 1.5, 8)[groups] + offset
    y = rng.poisson(np.exp(eta)) if kind == "poisson" else rng.binomial(1, special.expit(eta))
    return pd.DataFrame({"y": y, "g": groups, "off": offset})


def _count_deviance(kind, y, mu, weights):
    log_term = special.xlogy(y, y / mu)
    if kind == "poisson":
        return 2 * np.dot(weights, log_term - (y - mu))
    return 2 * np.dot(weights, log_term + special.xlogy(1 - y, (1 - y) / (1 - mu)))


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("kind", ["poisson", "binomial"])
def test_glmm_empty_fixed_laplace_matches_groupwise_modes(kind, native, monkeypatch):
    monkeypatch.setattr(laplace, "_HAS_RUST", native)
    data = _count_data(kind)
    family = Poisson() if kind == "poisson" else Binomial()
    weights = np.linspace(0.6, 1.8, len(data))
    matrices = build_model_matrices(
        parse_formula("y ~ 0 + (1 | g)"), data, weights=weights, offset=np.asarray(data.off)
    )
    scale = 0.65
    modes = []
    expected = 0.0
    for group in range(8):
        mask = np.asarray(data.g == group)
        y = np.asarray(data.y)[mask]
        offset = np.asarray(data.off)[mask]
        w = weights[mask]
        inverse = np.exp if kind == "poisson" else special.expit
        mode = optimize.brentq(
            lambda b, w, y, offset, inverse: np.dot(w, y - inverse(offset + b)) - b / scale**2,
            -10.0,
            10.0,
            args=(w, y, offset, inverse),
        )
        mu = inverse(offset + mode)
        deviance = _count_deviance(kind, y, mu, w)
        information = np.dot(w, mu if kind == "poisson" else mu * (1 - mu))
        expected += deviance + mode**2 / scale**2 + np.log1p(scale**2 * information)
        modes.append(mode)

    actual, beta, effects = laplace.laplace_deviance_fast(np.array([scale]), matrices, family)

    assert beta.shape == (0,)
    assert_allclose(effects, modes, rtol=1e-9, atol=1e-9)
    assert_allclose(actual, expected, rtol=1e-10)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("kind", ["poisson", "binomial"])
@pytest.mark.parametrize("random,nagq", [(False, 1), (True, 1), (True, 7)])
def test_glmm_no_fixed_effects_fit_predict_and_refit(kind, native, random, nagq, monkeypatch):
    monkeypatch.setattr(laplace, "_HAS_RUST", native)
    data = _count_data(kind)
    family = Poisson() if kind == "poisson" else Binomial()
    model = glmer(
        "y ~ 0 + (1 | g)" if random else "y ~ 0",
        data,
        family=family,
        offset="off",
        nAGQ=nagq,
    )

    assert model.converged
    assert np.isfinite(model.deviance)
    assert model.beta.shape == (0,)
    assert model.fixef() == {}
    assert model.vcov().shape == (0, 0)
    assert model.confint() == {}
    assert model.df_residual() == len(data)
    assert model.npar() == int(random)
    assert isinstance(model.summary(), str)
    assert_allclose(model.predict(type="link", re_form="NA"), data.off)
    assert_allclose(model.predict(data, offset="off"), model.fitted())
    prediction = model.predict(re_form="NA", interval="confidence")
    expected = family.link.inverse(np.asarray(data.off))
    assert_allclose(prediction.fit, expected)
    assert_array_equal(prediction.se_fit, np.zeros(len(data)))
    assert_allclose(prediction.lower, expected)
    assert_allclose(prediction.upper, expected)
    assert_array_equal(
        model.predict(pd.DataFrame(index=range(3)), type="link", re_form="NA"), np.zeros(3)
    )
    if random:
        newdata = pd.DataFrame({"g": [0, 99], "off": [0.7, -0.2]})
        prediction = model.predict(newdata, offset="off", allow_new_levels=True, type="link")
        assert_allclose(prediction, [model.u[0] + 0.7, -0.2])
    if not random:
        assert model.n_iter == 0
        assert_allclose(
            model.deviance, _count_deviance(kind, np.asarray(data.y), expected, np.ones(len(data)))
        )
    assert np.all(np.isfinite(model.hatvalues()))
    with np.errstate(divide="raise", invalid="raise"):
        assert np.all(np.isnan(model.cooks_distance()))
        assert np.all(np.isnan(model.influence()["cooks_d"]))
    refit = model.refit(np.asarray(data.y))
    assert refit.beta.shape == (0,)
    assert_allclose(refit.fitted(), model.fitted(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("cls", [LmerResult, GlmerResult])
def test_empty_fixed_covariance_and_cooks_distance_need_no_projection(cls):
    model = SimpleNamespace(matrices=SimpleNamespace(n_fixed=0, n_obs=9))

    assert cls.vcov(model).shape == (0, 0)
    assert_array_equal(cls.cooks_distance(model), np.full(9, np.nan))


@pytest.mark.parametrize("method", ["Satterthwaite", "Kenward-Roger"])
def test_empty_fixed_degrees_of_freedom_need_no_covariance_work(method):
    model = SimpleNamespace(matrices=SimpleNamespace(n_fixed=0))
    compute = ddf.satterthwaite_df if method == "Satterthwaite" else ddf.kenward_roger_df

    result = compute(model)

    assert result.df.shape == (0,)
    assert result.as_dict() == {}
    assert result.method == method


@pytest.mark.parametrize("method", sorted(optimizers.SCIPY_OPTIMIZERS) + ["nlminb"])
def test_no_covariance_parameters_require_one_evaluation(method, monkeypatch):
    calls = []

    def objective(theta):
        calls.append(theta.copy())
        return 3.5

    def unexpected_minimize(*args, **kwargs):
        raise AssertionError("a parameter-free objective needs no optimizer")

    monkeypatch.setattr(optimizers, "minimize", unexpected_minimize)
    start = np.empty(0)
    result = optimizers.run_optimizer(objective, start, method, [], callback=unexpected_minimize)

    assert len(calls) == result.nfev == 1
    assert result.success
    assert result.fun == 3.5
    assert result.nit == 0
    assert result.x.shape == result.jac.shape == (0,)
    assert result.x is not start


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_parameter_free_objective_is_not_reported_as_converged(value):
    result = optimizers.run_optimizer(lambda theta: value, np.empty(0), "L-BFGS-B", [])

    assert not result.success
    assert result.nfev == 1


def test_parameter_free_objective_still_rejects_unknown_optimizer():
    with pytest.raises(ValueError, match="Unknown optimizer"):
        optimizers.run_optimizer(lambda theta: 0.0, np.empty(0), "unknown", [])


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("reml", [False, True])
def test_no_random_effects_still_estimates_nonempty_fixed_effects(balanced_data, native, reml):
    weights = np.linspace(0.5, 2.0, len(balanced_data))
    design = np.column_stack([np.ones(len(balanced_data)), balanced_data.x])
    adjusted = np.asarray(balanced_data.y - balanced_data.off)
    expected = linalg.lstsq(np.sqrt(weights)[:, None] * design, np.sqrt(weights) * adjusted)[0]
    residuals = adjusted - design @ expected
    sigma2 = np.dot(weights, residuals**2) / (len(balanced_data) - (2 if reml else 0))

    model = lmer(
        "y ~ x",
        balanced_data,
        weights=weights,
        offset="off",
        REML=reml,
        control=lmerControl(use_rust=native),
    )

    assert model.converged
    assert model.n_iter == 0
    assert model.function_evals == 1
    assert_allclose(model.beta, expected, rtol=1e-12)
    assert_allclose(model.sigma**2, sigma2, rtol=1e-12)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("kind", ["poisson", "binomial"])
def test_glmm_no_random_effects_still_estimates_fixed_intercept(kind, native, monkeypatch):
    monkeypatch.setattr(laplace, "_HAS_RUST", native)
    data = _count_data(kind)
    family = Poisson() if kind == "poisson" else Binomial()
    inverse = np.exp if kind == "poisson" else special.expit
    expected = optimize.brentq(lambda b: np.sum(data.y - inverse(data.off + b)), -8, 8)

    model = glmer("y ~ 1", data, family=family, offset="off")

    assert model.converged
    assert model.n_iter == 0
    assert_allclose(model.beta, [expected], rtol=1e-9)
