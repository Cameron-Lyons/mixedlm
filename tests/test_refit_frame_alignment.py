"""Formula-based workflows must use the latest refitted response."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from mixedlm import cross_validate, families, glmer, glmerControl, lmer, lmerControl
from mixedlm.inference.drop1 import drop1_glmer, drop1_lmer
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import chi2


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("kind", ["lmer", "poisson", "binomial", "grouped"])
def test_refit_update_cv_and_drop1_use_current_response(kind, backend):
    rng = np.random.default_rng(617)
    group = np.repeat(np.arange(8), 10)
    x = rng.normal(size=len(group))
    z = rng.normal(size=len(group))
    offset = rng.normal(scale=0.2, size=len(group))
    prior = rng.uniform(0.7, 1.3, size=len(group))
    group_effect = rng.normal(scale=0.5, size=8)[group]
    original_eta = -0.2 + 0.3 * x + 0.2 * z + offset + group_effect
    new_eta = 0.4 + 0.8 * x - 0.6 * z + offset + group_effect
    trials = rng.integers(10, 100, size=len(group))
    if kind == "lmer":
        original_response = original_eta + rng.normal(scale=0.4, size=len(group))
        new_response = new_eta + rng.normal(scale=0.4, size=len(group))
    elif kind == "poisson":
        original_response = rng.poisson(np.exp(original_eta))
        new_response = rng.poisson(np.exp(new_eta))
    else:
        n_trials = trials if kind == "grouped" else 1
        original_response = rng.binomial(n_trials, 1 / (1 + np.exp(-original_eta)))
        new_response = rng.binomial(n_trials, 1 / (1 + np.exp(-new_eta)))

    original_data = pd.DataFrame(
        {"y": original_response, "trials": trials, "x": x, "z": z, "group": group}
    )
    current_data = original_data.assign(y=new_response)
    if backend == "polars":
        pl = pytest.importorskip("polars")
        original_data = pl.DataFrame(original_data.to_dict(orient="list"))
        current_data = pl.DataFrame(current_data.to_dict(orient="list"))
    response_formula = "y / trials" if kind == "grouped" else "y"
    formula = f"{response_formula} ~ x + z + (1 | group)"
    if kind == "lmer":
        fit_options = {
            "REML": False,
            "control": lmerControl(
                check_conv=False, check_singular=False, check_nlev_gtreq_5="ignore"
            ),
        }
        fitter = lmer
        dropper = drop1_lmer
    else:
        fit_options = {
            "family": families.Poisson() if kind == "poisson" else families.Binomial(),
            "control": glmerControl(
                check_conv=False, check_singular=False, check_nlev_gtreq_5="ignore"
            ),
        }
        fitter = glmer
        dropper = drop1_glmer

    original = fitter(formula, original_data, weights=prior, offset=offset, **fit_options)
    # Populate the original likelihood cache before cloning the response.
    original_loglik = original.logLik().value
    refitted = original.refit(new_response)
    fresh = fitter(formula, current_data, weights=prior, offset=offset, **fit_options)
    assert_array_equal(refitted.model_frame()["y"].to_numpy(), new_response)
    assert_array_equal(original.model_frame()["y"].to_numpy(), original_response)
    assert original.logLik().value == original_loglik
    assert_allclose(refitted.beta, fresh.beta, rtol=1e-5, atol=1e-5)
    assert refitted.logLik().value == pytest.approx(fresh.logLik().value, rel=1e-7)

    updated = refitted.update(control=fit_options["control"])
    assert_array_equal(updated.matrices.y, fresh.matrices.y)
    assert_allclose(updated.matrices.weights, fresh.matrices.weights)
    assert_array_equal(updated.matrices.offset, offset)
    assert_allclose(updated.beta, fresh.beta, rtol=1e-7, atol=1e-7)
    assert_allclose(updated.theta, fresh.theta, rtol=1e-7, atol=1e-7)
    assert updated.logLik().value == pytest.approx(fresh.logLik().value, rel=1e-9)

    # Two folds leave four training groups; the control above omits the level-count advice.
    cv_options = {
        "cv": 2,
        "group": "group",
        "random_state": 35,
        "metrics": "mse",
        "fit_kwargs": {"control": fit_options["control"]},
    }
    refitted_cv = cross_validate(refitted, **cv_options)
    fresh_cv = cross_validate(fresh, **cv_options)
    assert_allclose(refitted_cv.predictions, fresh_cv.predictions, rtol=1e-9, atol=1e-9)
    assert refitted_cv.scores == pytest.approx(fresh_cv.scores, rel=1e-9)

    deletions = dropper(refitted, refitted.model_frame())
    assert deletions.terms == ["x", "z"]
    assert deletions.full_model_aic == pytest.approx(fresh.AIC(), rel=1e-7)
    for index, retained in enumerate(["z", "x"]):
        reduced = fitter(
            f"{response_formula} ~ {retained} + (1 | group)",
            current_data,
            weights=prior,
            offset=offset,
            start=refitted.theta,
            **fit_options,
        )
        lrt = max(0, 2 * (fresh.logLik().value - reduced.logLik().value))
        assert deletions.aic[index] == pytest.approx(reduced.AIC(), rel=1e-9)
        assert deletions.lrt[index] == pytest.approx(lrt, rel=1e-6)
        assert deletions.p_value[index] == pytest.approx(chi2.sf(lrt, 1), rel=1e-6)


def test_glmm_update_and_drop1_keep_the_fitted_control():
    rng = np.random.default_rng(0)
    data = pd.DataFrame(
        {
            "y": rng.binomial(1, 0.5, 80),
            "x": rng.normal(size=80),
            "group": np.repeat(np.arange(8), 10),
        }
    )
    control = glmerControl(check_singular=False)
    model = glmer("y ~ x + (1 | group)", data, family=families.Binomial(), control=control)
    assert_array_equal(model.theta, [0.0])
    control.check_singular = True

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert not model.update()._refit_control().check_singular
        assert not model.refit()._refit_control().check_singular
        drop1_glmer(model, data)
    assert [str(warning.message) for warning in caught] == []


@pytest.mark.parametrize("generalized", [False, True], ids=["lmer", "glmer"])
def test_formula_refits_keep_the_fitted_contrast_basis(generalized):
    rng = np.random.default_rng(164)
    groups = np.repeat(np.arange(8), 18)
    arms = np.tile(["a", "b", "c"], len(groups) // 3)
    x = rng.normal(size=len(groups))
    z = rng.normal(size=len(groups))
    slopes = np.select([arms == "a", arms == "b"], [0.7, -0.1], default=-0.6)
    eta = 0.5 + rng.normal(scale=0.45, size=8)[groups] + x * slopes + 0.15 * z
    y = rng.poisson(np.exp(eta)) if generalized else eta + rng.normal(scale=0.3, size=len(groups))
    data = pd.DataFrame({"y": y, "x": x, "z": z, "arm": arms, "group": groups})
    fit = glmer if generalized else lmer
    options = {"family": families.Poisson(), "nAGQ": 0} if generalized else {"REML": False}
    # Custom contrasts give arm and x:arm their own basis; refits must reuse it.
    formula = "y ~ x * arm + z + (1 | group)"
    contrasts = {"arm": np.array([[1, 0], [0, 1], [-1, -1]], dtype=float)}
    full = fit(formula, data, contrasts=contrasts, method="COBYQA", **options)

    updated = full.update(method="COBYQA")
    optimizer_refit = full.allFit(data, ["COBYQA"]).fits["COBYQA"]
    for refit in (updated, optimizer_refit):
        assert_array_equal(refit.matrices.X, full.matrices.X)
        assert_allclose(refit.beta, full.beta, rtol=1e-9)
    overridden = full.update(contrasts={"arm": "treatment"})
    assert not np.array_equal(overridden.matrices.X, full.matrices.X)

    deletions = full.drop1(data)
    reduced = fit(
        "y ~ x * arm + (1 | group)", data, contrasts=contrasts, start=full.theta, **options
    )
    i = deletions.terms.index("z")
    assert deletions.aic[i] == pytest.approx(reduced.AIC(), rel=1e-8)
    assert deletions.lrt[i] == pytest.approx(2 * (full.logLik().value - reduced.logLik().value))


@pytest.mark.parametrize("generalized", [False, True], ids=["lmer", "glmer"])
def test_refits_on_original_rows_keep_full_models_omissions(generalized):
    rng = np.random.default_rng(683)
    groups = np.repeat(np.arange(8), 10)
    x = rng.normal(size=len(groups))
    z = rng.normal(size=len(groups))
    eta = 0.4 + 0.35 * x + 0.2 * z + rng.normal(scale=0.5, size=8)[groups]
    y = rng.poisson(np.exp(eta)) if generalized else eta + rng.normal(scale=0.3, size=len(groups))
    data = pd.DataFrame({"y": y, "x": x, "z": z, "g": groups})
    weights = rng.uniform(0.7, 1.3, size=len(data))
    offset = rng.normal(scale=0.1, size=len(data))
    data.loc[2, "z"] = weights[5] = offset[9] = np.nan
    keep = np.ones(len(data), dtype=bool)
    keep[[2, 5, 9]] = False
    fit = glmer if generalized else lmer
    options = {"family": families.Poisson(), "nAGQ": 0} if generalized else {"REML": False}
    full = fit(
        "y ~ x + z + (1 | g)",
        data,
        weights=weights,
        offset=offset,
        na_action="exclude",
        **options,
    )
    reduced = fit(
        "y ~ x + (1 | g)",
        data[keep],
        weights=weights[keep],
        offset=offset[keep],
        start=full.theta,
        **options,
    )
    for frame in (data, full.model_frame()):
        deletion = full.drop1(frame)
        assert deletion.aic[deletion.terms.index("z")] == pytest.approx(reduced.AIC(), rel=1e-8)
        alternative = full.allFit(frame, ["COBYQA"]).fits["COBYQA"]
        assert alternative.nobs() == keep.sum()
        assert_allclose(alternative.matrices.weights, weights[keep])
        assert_allclose(alternative.matrices.offset, offset[keep])
        if frame is data:
            assert_array_equal(np.isnan(alternative.fitted()), ~keep)
