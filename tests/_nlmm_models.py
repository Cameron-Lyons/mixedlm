"""Nonlinear mixed-model data, fits and unfitted results shared by nlmer tests."""

from itertools import product

import numpy as np
import pandas as pd
from mixedlm import nlme, nlmer
from mixedlm.models.nlmer import NlmerResult
from mixedlm.nlme.models import SSasymp


def create_nlme_data(n_groups: int = 8, n_per_group: int = 10, seed: int = 42) -> pd.DataFrame:
    """Asymptotic growth curves with three random parameters per subject."""
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
    """Asymptotic growth curves with small subject effects."""
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


class PythonAsymptotic(SSasymp):
    """Importable custom model using the Python estimator in worker processes."""


def fitted_model(custom=False):
    """A weighted, offset SSasymp fit with a random asymptote."""
    rng = np.random.default_rng(17)
    n = 40
    x = np.tile(np.linspace(0, 10, 10), 4)
    weights = np.linspace(1.0, 4.0, n)
    offsets = np.linspace(-2.0, 2.0, n)
    y = 10.0 + (3.0 - 10.0) * np.exp(-np.exp(-1.0) * x)
    y += np.repeat(rng.normal(0, 0.5, 4), 10) + offsets + rng.normal(0, 0.2, n)
    data = pd.DataFrame({"x": x, "y": y, "subject": np.repeat(list("abcd"), 10)})
    return nlmer(
        PythonAsymptotic() if custom else SSasymp(),
        data,
        x_var="x",
        y_var="y",
        group_var="subject",
        weights=weights,
        offset=offsets,
        random_params=["Asym"],
        pnls_maxiter=2000,
    )


def fit_asymptotic_nlmm():
    """Fit SSasymp with three correlated random parameters that the data identify."""
    rng = np.random.default_rng(814)
    times = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0])
    # Balanced contrasts give all three random parameters independent variation
    # while preserving a nonzero Asym/R0 covariance across enough groups.
    contrasts = np.tile(np.array(list(product([-1.0, 1.0], repeat=3))), (3, 1))
    effects = contrasts @ np.array([[0.8, 0.1, 0.02], [0.0, 0.3, 0.01], [0.0, 0.0, 0.08]])
    rows = []
    for group_index, (asym_effect, r0_effect, lrc_effect) in enumerate(effects):
        asym = 12.0 + asym_effect
        r0 = 2.0 + r0_effect
        lrc = -1.0 + lrc_effect
        # Observe both the transition and plateau with a strong parameter
        # contrast so rate and asymptote uncertainty remain identifiable.
        for time in times:
            response = asym - (asym - r0) * np.exp(-np.exp(lrc) * time)
            rows.append(
                {
                    "subject": f"S{group_index + 1}",
                    "time": time,
                    "response": response + rng.normal(0.0, 0.2),
                }
            )
    data = pd.DataFrame(rows)
    result = nlmer(
        nlme.SSasymp(),
        data,
        x_var="time",
        y_var="response",
        group_var="subject",
        start={"Asym": 12.0, "R0": 2.0, "lrc": -1.0},
    )
    assert result.converged and result.pnls_converged
    np.testing.assert_allclose(result.phi, [12.0, 2.0, -1.0], atol=0.1, rtol=0)
    variances = np.diag(result.vcov())
    assert np.all(np.isfinite(variances)) and np.all(variances > 0)
    return result


def make_result(random_params=(0, 1), n_groups=4, per_group=9):
    """An unfitted SSasymp result with the given random parameters."""
    n = n_groups * per_group
    order = np.random.default_rng(13).permutation(n)
    q = len(random_params)
    factor = np.tril(np.full((q, q), 0.15))
    np.fill_diagonal(factor, 0.6)
    return NlmerResult(
        model=SSasymp(),
        group_var="subject",
        phi=np.array([10.0, 3.0, -1.0]),
        theta=factor[np.tril_indices(q)],
        sigma=0.3,
        b=np.full((n_groups, q), 0.5),
        random_params=list(random_params),
        deviance=0.0,
        converged=True,
        n_iter=1,
        x=np.tile(np.linspace(0, 8, per_group), n_groups)[order],
        y=np.zeros(n),
        groups=np.repeat(np.arange(n_groups), per_group)[order],
        group_levels=[f"g{i}" for i in range(n_groups)],
        _weights=np.linspace(0.5, 3.0, n),
        _offset=np.linspace(-2.0, 2.0, n),
    )


def legacy_draws(result, count, seed, include_re=True):
    """Independent reference for the previous seeded nonlinear simulation."""
    rng = np.random.RandomState(seed)
    q = len(result.random_params)
    factor = np.zeros((q, q))
    factor[np.tril_indices(q)] = result.theta
    covariance = factor @ factor.T * result.sigma**2 + 1e-8 * np.eye(q)
    draws = np.zeros((len(result.y), count))
    for draw in range(count):
        effects = (
            rng.multivariate_normal(np.zeros(q), covariance, size=len(result.group_levels))
            if include_re and q
            else np.zeros_like(result.b)
        )
        mean = np.zeros(len(result.y))
        for group in range(len(result.group_levels)):
            rows = result.groups == group
            params = result.phi.copy()
            for column, parameter in enumerate(result.random_params):
                params[parameter] += effects[group, column]
            mean[rows] = result.model.predict(params, result.x[rows])
        mean += result.offset(copy=False)
        draws[:, draw] = mean + rng.standard_normal(len(mean)) * (
            result.sigma / np.sqrt(result.weights(copy=False))
        )
    return draws[:, 0] if count == 1 else draws
