"""Formula-based workflows must use the latest refitted response."""

from __future__ import annotations

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
            "control": lmerControl(check_conv=False, check_singular=False),
        }
        fitter = lmer
        dropper = drop1_lmer
    else:
        fit_options = {
            "family": families.Poisson() if kind == "poisson" else families.Binomial(),
            "control": glmerControl(check_conv=False, check_singular=False),
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
