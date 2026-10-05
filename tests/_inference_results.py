"""Fitted models and hand-built inference results shared by inference tests."""

import numpy as np
import pandas as pd
from mixedlm import families, glmer, lmer
from mixedlm.inference.bootstrap import BootstrapResult
from mixedlm.inference.emmeans import EmmeanResult, Emmeans


def fit_random_intercept_lmm():
    """Sixteen groups with a large random intercept and two predictors."""
    rng = np.random.default_rng(20260803)
    n_groups = 16
    observations_per_group = 12
    group_index = np.repeat(np.arange(n_groups), observations_per_group)
    x = rng.normal(size=len(group_index))
    z = rng.normal(size=len(group_index))
    random_intercept = rng.normal(scale=2.0, size=n_groups)
    y = (
        1.5
        + 0.8 * x
        - 0.35 * z
        + random_intercept[group_index]
        + rng.normal(scale=0.3, size=len(group_index))
    )
    data = pd.DataFrame({"y": y, "x": x, "z": z, "group": [f"G{value}" for value in group_index]})
    return lmer("y ~ x + z + (1 | group)", data)


def fit_random_intercept_glmm():
    """A logistic random-intercept fit of twenty groups."""
    rng = np.random.default_rng(20260804)
    n_groups = 20
    observations_per_group = 12
    group_index = np.repeat(np.arange(n_groups), observations_per_group)
    x = rng.normal(size=len(group_index))
    random_intercept = rng.normal(scale=0.55, size=n_groups)
    eta = -0.4 + 0.75 * x + random_intercept[group_index]
    probability = 1 / (1 + np.exp(-eta))
    y = rng.binomial(1, probability)
    data = pd.DataFrame({"y": y, "x": x, "group": [f"G{value}" for value in group_index]})
    return glmer("y ~ x + (1 | group)", data, family=families.Binomial())


def bootstrap_with_failures() -> BootstrapResult:
    """Five bootstrap samples; the last failed and holds non-finite values."""
    return BootstrapResult(
        n_boot=5,
        beta_samples=np.array(
            [
                [0.8, 1.7],
                [0.9, 1.9],
                [1.1, 2.2],
                [1.3, 2.4],
                [np.nan, 2.1],
            ]
        ),
        theta_samples=np.array([[0.3], [0.4], [0.5], [0.6], [np.inf]]),
        sigma_samples=np.array([0.8, 0.9, 1.0, 1.1, 1.2]),
        fixed_names=["intercept", "slope"],
        original_beta=np.array([1.0, 2.0]),
        original_theta=np.array([0.45]),
        original_sigma=1.0,
        n_failed=1,
    )


def identity_emmeans(df=17.0, n=4):
    """Marginal means with an identity coefficient matrix and known covariance."""
    beta = np.linspace(-0.7, 1.0, n)
    covariance = np.diag(np.linspace(0.2, 0.5, n)) + 0.08
    zeros = np.zeros(n)
    return Emmeans(
        EmmeanResult(beta, zeros, df, zeros, zeros, pd.DataFrame({"group": list(range(n))}), 0.95),
        np.eye(n),
        covariance,
        beta,
        df,
        ["group"],
        [list(range(n))],
    )


def synthetic_emmeans(n_levels: int = 12, n_beta: int = 5) -> Emmeans:
    """Marginal means of random linear combinations of a known coefficient vector."""
    rng = np.random.default_rng(91)
    coefficients = rng.normal(size=(n_levels, n_beta))
    covariance_factor = rng.normal(size=(n_beta, n_beta))
    covariance = covariance_factor @ covariance_factor.T
    beta = rng.normal(size=n_beta)
    estimates = coefficients @ beta
    zeros = np.zeros(n_levels)
    result = EmmeanResult(
        emmean=estimates,
        se=zeros,
        df=80.0,
        lower=zeros,
        upper=zeros,
        grid=pd.DataFrame({"treatment": [f"L{i}" for i in range(n_levels)]}),
        level=0.95,
    )
    return Emmeans(
        result=result,
        _L=coefficients,
        _vcov=covariance,
        _beta=beta,
        _df=80.0,
        _specs=["treatment"],
        _levels=[list(range(n_levels))],
    )
