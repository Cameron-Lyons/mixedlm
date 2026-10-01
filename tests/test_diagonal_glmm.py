"""Independent scalar integrals check the native diagonal random-effect solve."""

import numpy as np
import pytest
from mixedlm import _rust
from numpy.testing import assert_allclose
from scipy import optimize, sparse, special


@pytest.mark.parametrize("groups", [1, 8, 127, 128])
@pytest.mark.parametrize("slope", [False, True])
@pytest.mark.parametrize("family", ["poisson", "binomial"])
@pytest.mark.parametrize("order", [1, 9])
def test_independent_scalar_modes_and_integrals(groups, slope, family, order):
    rng = np.random.default_rng(417)
    rows = np.arange(6 * groups)
    group = rows % groups
    values = rng.uniform(-1.2, 1.2, len(rows)) if slope else np.ones(len(rows))
    # Zero rows, duplicate CSC entries, and one unused level remain valid.
    values[::17] = 0.0
    design = sparse.coo_matrix((values, (rows, group)), shape=(len(rows), groups + 1)).tocsc()
    design = sparse.csc_matrix(
        (np.repeat(design.data / 2, 2), np.repeat(design.indices, 2), 2 * design.indptr),
        shape=design.shape,
    )
    offset = 0.15 + 0.2 * np.sin(rows)
    weights = 0.5 + (rows % 7) / 4
    eta = offset + values * 0.3 * np.cos(group)
    if family == "poisson":
        y = rng.poisson(np.exp(eta)).astype(float)
        inverse = np.exp
    else:
        trials = 3 + rows % 5
        y = rng.binomial(trials, special.expit(eta)) / trials
        weights *= trials
        inverse = special.expit
    theta = 0.65
    expected_random = np.zeros(groups + 1)
    expected = 0.0
    nodes, node_weights = np.polynomial.hermite.hermgauss(order)
    for level in range(groups):
        keep = group == level
        z, w, yy, off = values[keep], weights[keep], y[keep], offset[keep]
        mode = optimize.brentq(
            lambda u, zz=z, ww=w, response=yy, oo=off: u
            - theta * np.dot(ww * zz, response - inverse(oo + theta * zz * u)),
            -30,
            30,
        )
        expected_random[level] = theta * mode
        mu = inverse(off + theta * z * mode)
        variance = mu if family == "poisson" else mu * (1 - mu)
        precision = 1 + theta**2 * np.dot(w * z**2, variance)
        spherical = mode + np.sqrt(2 / precision) * nodes
        linear = off[None, :] + theta * spherical[:, None] * z
        mean = inverse(linear)
        if family == "poisson":
            unit_deviance = 2 * (special.xlogy(yy, yy) - yy * linear - yy + mean)
        else:
            unit_deviance = 2 * (
                special.xlogy(yy, yy)
                + special.xlogy(1 - yy, 1 - yy)
                - yy * linear
                + np.logaddexp(0, linear)
            )
        log_integrand = -0.5 * (unit_deviance @ w + spherical**2) + nodes**2
        expected += np.log(precision * np.pi) - 2 * special.logsumexp(
            np.log(node_weights) + log_integrand
        )

    actual, beta, random, converged = _rust.glmm_deviance(
        y,
        np.empty((len(rows), 0)),
        design.data,
        design.indices.astype(np.int64),
        design.indptr.astype(np.int64),
        design.shape,
        weights,
        offset,
        np.array([theta]),
        [groups + 1],
        [1],
        [True],
        family,
        "log" if family == "poisson" else "logit",
        order,
        tol=1e-10,
    )
    assert converged
    assert beta == []
    assert_allclose(random, expected_random, atol=2e-10, rtol=2e-10)
    assert_allclose(actual, expected, atol=2e-9, rtol=2e-12)
