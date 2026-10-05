"""Influence diagnostics on models without fixed effects or with excluded rows."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from mixedlm import diagnostics, families, glmer, lmer, load_cbpp, load_sleepstudy
from numpy.testing import assert_allclose


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
def test_models_without_fixed_effects_report_nan_cooks_distance(kind) -> None:
    if kind == "lmm":
        result = lmer("Reaction ~ 0 + (Days | Subject)", load_sleepstudy())
    else:
        result = glmer("incidence / size ~ 0 + (1 | herd)", load_cbpp(), family=families.Binomial())

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        method = result.cooks_distance()
        helper = diagnostics.cooks_distance(result)
        influence = result.influence()

    assert method.shape == (result.nobs(),)
    assert np.isnan(method).all()
    assert np.isnan(helper).all()
    assert np.isnan(influence["cooks_d"]).all()
    assert_allclose(influence["hat"], result.hatvalues())


def test_glmm_influence_with_excluded_rows_covers_the_fitted_observations() -> None:
    data = load_cbpp()
    data.loc[[3, 10], "incidence"] = np.nan
    result = glmer(
        "incidence / size ~ period + (1 | herd)",
        data,
        family=families.Binomial(),
        na_action="exclude",
    )
    n_fit = result.nobs()
    pearson = result.residuals(type="pearson", na_expand=False)
    h = result.hatvalues()
    p = result.matrices.n_fixed

    cooks = result.cooks_distance()
    influence = result.influence()

    assert n_fit == len(data) - 2
    assert len(result.residuals()) == len(data)
    assert_allclose(cooks, pearson**2 / p * h / (1 - h) ** 2, rtol=1e-12)
    assert {name: len(values) for name, values in influence.items()} == {
        "hat": n_fit,
        "cooks_d": n_fit,
        "pearson_resid": n_fit,
        "deviance_resid": n_fit,
    }
    assert_allclose(influence["pearson_resid"], pearson)
    assert_allclose(influence["deviance_resid"], result.residuals(na_expand=False))


def test_lmm_influence_dictionary_matches_its_definitions() -> None:
    data = load_sleepstudy()
    weights = np.linspace(0.5, 2.0, len(data))
    result = lmer("Reaction ~ Days + (Days | Subject)", data, weights=weights)
    h = result.hatvalues()
    r = np.sqrt(weights) * result.residuals()
    n, p = result.nobs(), result.matrices.n_fixed
    loo_variance = (np.sum(r**2) - r**2 / (1 - h)) / (n - p - 1)

    influence = result.influence()

    assert_allclose(influence["hat"], h)
    assert_allclose(influence["cooks_d"], r**2 / (p * result.sigma**2) * h / (1 - h) ** 2)
    assert_allclose(influence["std_resid"], r / (result.sigma * np.sqrt(1 - h)))
    assert_allclose(influence["student_resid"], r / np.sqrt(loo_variance * (1 - h)))
