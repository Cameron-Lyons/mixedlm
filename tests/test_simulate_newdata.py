"""simulate(newdata=, newparams=) follows lme4's simulate.merMod."""

import warnings

import numpy as np
import pandas as pd
import pytest
from mixedlm import families, glmer, lmer
from numpy.testing import assert_allclose, assert_array_equal

from tests._datasets import CBPP, SLEEPSTUDY


@pytest.fixture(scope="module")
def sleep_fit():
    return lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY)


@pytest.fixture(scope="module")
def cbpp_fit():
    return glmer(
        "cbind(incidence, size - incidence) ~ period + (1 | herd)",
        CBPP,
        family=families.Binomial(),
    )


def test_fitted_rows_reproduce_the_default_draws(sleep_fit):
    for nsim in (1, 3):
        assert_array_equal(
            sleep_fit.simulate(nsim=nsim, seed=4, newdata=SLEEPSTUDY),
            sleep_fit.simulate(nsim=nsim, seed=4),
        )


def test_new_groups_and_parameters_set_the_moments(sleep_fit):
    n_groups = 3000
    data = pd.DataFrame(
        {"Days": np.zeros(n_groups), "Subject": [f"new{i}" for i in range(n_groups)]}
    )
    draws = sleep_fit.simulate(
        seed=1,
        newdata=data,
        newparams={"beta": [100.0, 5.0], "theta": [1.5, 0.0, 0.0], "sigma": 2.0},
    )
    # Intercept SD theta * sigma = 3 and residual SD 2 give variance 13.
    assert draws.mean() == pytest.approx(100.0, abs=0.3)
    assert draws.var() == pytest.approx(13.0, rel=0.08)


def test_parameters_alone_and_named_coefficients(sleep_fit):
    named = sleep_fit.simulate(
        seed=2, newparams={"beta": {"Days": 1.0, "(Intercept)": 0.0}}, use_re=False
    )
    positional = sleep_fit.simulate(seed=2, newparams={"beta": [0.0, 1.0]}, use_re=False)
    assert_array_equal(named, positional)
    residual = positional - SLEEPSTUDY["Days"].to_numpy()
    assert residual.std() == pytest.approx(sleep_fit.sigma, rel=0.15)


def test_new_rows_without_random_effects_need_no_groups(sleep_fit):
    draws = sleep_fit.simulate(
        nsim=2000, seed=3, newdata=pd.DataFrame({"Days": [1.0, 2.0]}), re_form="~0"
    )
    assert draws.shape == (2, 2000)
    assert_allclose(
        draws.mean(axis=1), sleep_fit.beta[0] + sleep_fit.beta[1] * np.array([1.0, 2.0]), rtol=0.01
    )


def test_prior_weights_scale_residual_variance(sleep_fit):
    data = pd.DataFrame({"Days": np.zeros(4000), "Subject": "308", "w": 4.0})
    draws = sleep_fit.simulate(
        seed=5, newdata=data, weights="w", newparams={"theta": [0.0, 0.0, 0.0]}
    )
    assert draws.std() == pytest.approx(sleep_fit.sigma / 2.0, rel=0.05)


def test_binomial_counts_read_trials_from_new_data(cbpp_fit):
    data = CBPP.assign(herd="other" + CBPP["herd"].astype(str))
    draws = cbpp_fit.simulate(nsim=50, seed=6, newdata=data)
    trials = CBPP["size"].to_numpy()[:, None]
    assert np.all((draws >= 0) & (draws <= trials))
    assert np.all(draws == np.round(draws))

    with pytest.raises(ValueError, match="missing binomial trial variable"):
        glmer("incidence / size ~ period + (1 | herd)", CBPP, family=families.Binomial()).simulate(
            newdata=CBPP.drop(columns="size")
        )


def test_formula_offsets_are_evaluated_on_new_data():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        data = pd.DataFrame(
            {"y": [1.0, 3.0, 2.0, 5.0, 4.0, 6.0], "t": 1.0, "g": ["a", "a", "b", "b", "c", "c"]}
        )
        fit = glmer("y ~ 1 + offset(log(t)) + (1 | g)", data, family=families.Poisson())
    draws = fit.simulate(nsim=4000, seed=7, newdata=data.assign(t=10.0), use_re=False)
    assert draws.mean() == pytest.approx(10.0 * np.exp(fit.beta[0]), rel=0.03)


@pytest.mark.parametrize(
    ("newparams", "message"),
    [
        ({"beta": [1.0]}, "must hold 2 finite values"),
        ({"beta": {"Days": 1.0}}, "exactly the coefficients"),
        ({"theta": [np.nan, 0.0, 1.0]}, "finite values"),
        ({"sigma": 0.0}, "positive finite"),
        ({"sigma": True}, "positive finite"),
        ({"gamma": 1.0}, "Unknown newparams"),
    ],
)
def test_invalid_parameters_are_rejected(sleep_fit, newparams, message):
    with pytest.raises(ValueError, match=message):
        sleep_fit.simulate(newparams=newparams)


def test_sigma_is_not_a_glmm_parameter_and_weights_need_new_data(sleep_fit, cbpp_fit):
    with pytest.raises(ValueError, match="Unknown newparams"):
        cbpp_fit.simulate(newparams={"sigma": 1.0})
    with pytest.raises(ValueError, match="only be supplied with newdata"):
        sleep_fit.simulate(weights=2.0)
    with pytest.raises(TypeError, match="mapping"):
        sleep_fit.simulate(newparams=[1.0, 2.0])
