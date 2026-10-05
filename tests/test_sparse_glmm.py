"""Sparse native GLMM systems retain the full penalized model algebra."""

import numpy as np
import pytest
from mixedlm import _rust
from numpy.testing import assert_allclose, assert_array_equal
from scipy import linalg, sparse


def _native_laplace(args):
    """Evaluate the one-shot and prepared native entry points, which must agree exactly."""
    actual = _rust.glmm_deviance(*args, 1)
    prepared = _rust.GlmmProblem(*args[:8], *args[9:]).evaluate(args[8])
    for value, reference in zip(prepared, actual, strict=True):
        assert_array_equal(value, reference)
    return actual


def _fixture(layout, p, zero, noncanonical):
    rng = np.random.default_rng(347)
    n = 768
    rows = np.arange(n)
    slope = rng.uniform(-1, 1, n)
    if layout in {"correlated", "diagonal", "mixed"}:
        levels, terms = [80], [2]
        correlated = [layout != "diagonal"]
        factors = [np.array([[0.6, 0.0], [0.15 if correlated[0] else 0.0, 0.35]])]
        indices = (2 * (rows % 80)[:, None] + np.arange(2)).ravel()
        values = np.column_stack((np.ones(n), slope)).ravel()
        z = sparse.coo_matrix((values, (np.repeat(rows, 2), indices)), shape=(n, 160)).tocsc()
        if layout == "mixed":
            levels.append(72)
            terms.append(1)
            correlated.append(True)
            factors.append(np.array([[0.4]]))
            extra = sparse.coo_matrix((np.ones(n), (rows, rng.integers(0, 72, n))), shape=(n, 72))
            z = sparse.hstack((z, extra)).tocsc()
    elif layout == "crossed":
        levels, terms, correlated = [96, 64], [1, 1], [True, True]
        factors = [np.array([[0.5]]), np.array([[0.35]])]
        z = sparse.coo_matrix(
            (
                np.ones(2 * n),
                (
                    np.repeat(rows, 2),
                    np.column_stack((rows % 96, 96 + rng.integers(0, 64, n))).ravel(),
                ),
            ),
            shape=(n, 160),
        ).tocsc()
    else:
        levels, terms, correlated = [160], [1], [True]
        factors = [np.array([[0.5]])]
        if layout == "dense":
            z = sparse.csc_matrix(rng.normal(scale=0.08, size=(n, 160)))
        elif layout == "star":
            indices = np.column_stack((np.zeros(n, dtype=int), 1 + rows % 159)).ravel()
            z = sparse.coo_matrix(
                (np.ones(2 * n), (np.repeat(rows, 2), indices)), shape=(n, 160)
            ).tocsc()
        elif layout == "overlap":
            indices = np.column_stack((rows % 160, (rows + 17 + rows // 160) % 160)).ravel()
            z = sparse.coo_matrix(
                (np.column_stack((np.ones(n), slope)).ravel(), (np.repeat(rows, 2), indices)),
                shape=(n, 160),
            ).tocsc()
        else:
            # Keep four unused levels in the design.
            z = sparse.coo_matrix((np.ones(n), (rows, rows % 156)), shape=(n, 160)).tocsc()
    if zero:
        factors = [np.zeros_like(factor) for factor in factors]
    theta = np.concatenate(
        [
            factor[np.tril_indices(width)] if corr else np.diag(factor)
            for factor, width, corr in zip(factors, terms, correlated, strict=True)
        ]
    )
    covariance = linalg.block_diag(
        *(np.kron(np.eye(count), factor) for count, factor in zip(levels, factors, strict=True))
    )
    if noncanonical:
        data, indices, offsets = [], [], [0]
        for column in range(z.shape[1]):
            for entry in range(z.indptr[column + 1] - 1, z.indptr[column] - 1, -1):
                indices.extend([z.indices[entry], z.indices[entry]])
                data.extend([0.25 * z.data[entry], 0.75 * z.data[entry]])
            offsets.append(len(data))
        z = sparse.csc_matrix((data, indices, offsets), shape=z.shape)
    x = rng.normal(size=(n, p))
    if p:
        x[:, 0] = 1.0
    offset = 0.1 * np.sin(rows)
    weights = np.linspace(0.4, 2.0, n)
    y = offset + x @ rng.normal(scale=0.3, size=p)
    y += z @ rng.normal(scale=0.2, size=z.shape[1]) + rng.normal(scale=0.3, size=n)
    args = (
        y,
        x,
        z.data,
        z.indices.astype(np.int64),
        z.indptr.astype(np.int64),
        z.shape,
        weights,
        offset,
        theta,
        levels,
        terms,
        correlated,
        "gaussian",
        "identity",
    )
    return args, z.toarray(), covariance


@pytest.mark.parametrize(
    "layout",
    ["intercept", "correlated", "diagonal", "crossed", "overlap", "mixed", "dense", "star"],
)
@pytest.mark.parametrize("p", [0, 3])
@pytest.mark.parametrize("zero", [False, True])
@pytest.mark.parametrize("noncanonical", [False, True])
def test_native_sparse_system_matches_full_joint_gaussian_solution(layout, p, zero, noncanonical):
    args, z, covariance = _fixture(layout, p, zero, noncanonical)
    y, x, weights, offset = args[0], args[1], args[6], args[7]
    scaled_z = z @ covariance
    design = np.column_stack((x, scaled_z))
    information = design.T @ (weights[:, None] * design)
    information[p:, p:] += np.eye(z.shape[1])
    solution = np.linalg.solve(information, design.T @ (weights * (y - offset)))
    expected_beta = solution[:p]
    expected_random = covariance @ solution[p:]
    residual = y - offset - design @ solution
    expected_deviance = np.dot(weights * residual, residual) + np.dot(solution[p:], solution[p:])
    logdet = np.linalg.slogdet(information[p:, p:])[1]

    actual, beta, random, converged = _native_laplace(args)
    assert converged
    assert actual == pytest.approx(expected_deviance + logdet, rel=1e-11, abs=1e-10)
    assert_allclose(beta, expected_beta, rtol=1e-10, atol=2e-11)
    assert_allclose(random, expected_random, rtol=1e-10, atol=2e-11)


@pytest.mark.parametrize("groups", [127, 128, 129, 20_000])
def test_random_intercept_scaling_matches_closed_form(groups):
    n = 3 * groups
    rows = np.arange(n)
    group = rows % groups
    weights = 0.5 + (rows % 7) / 4.0
    offset = 0.1 * np.cos(rows)
    y = 0.3 * np.sin(group) + offset + 0.1 * np.sin(rows / groups)
    theta = 0.65
    z = sparse.coo_matrix((np.ones(n), (rows, group)), shape=(n, groups)).tocsc()
    information = 1 + theta**2 * np.bincount(group, weights=weights, minlength=groups)
    spherical = (
        theta * np.bincount(group, weights=weights * (y - offset), minlength=groups) / information
    )
    expected_random = theta * spherical
    residual = y - offset - expected_random[group]
    expected = np.dot(weights * residual, residual) + np.dot(spherical, spherical)
    expected += np.log(information).sum()

    actual, beta, random, converged = _native_laplace(
        (
            y,
            np.empty((n, 0)),
            z.data,
            z.indices.astype(np.int64),
            z.indptr.astype(np.int64),
            z.shape,
            weights,
            offset,
            np.array([theta]),
            [groups],
            [1],
            [True],
            "gaussian",
            "identity",
        )
    )

    assert converged
    assert len(beta) == 0
    assert_allclose(random, expected_random, rtol=1e-12, atol=1e-12)
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-9)


@pytest.mark.parametrize("family", ["poisson", "binomial"])
@pytest.mark.parametrize(
    "layout",
    ["intercept", "correlated", "diagonal", "crossed", "overlap", "mixed", "dense", "star"],
)
@pytest.mark.parametrize("zero", [False, True])
def test_sparse_nongaussian_likelihood_matches_dense_optimizer(family, layout, zero):
    from scipy import optimize, special

    args, z, covariance = _fixture(layout, 3, zero, True)
    x, weights, offset = args[1], args[6], args[7]
    design = np.column_stack((x, z @ covariance))
    p = x.shape[1]
    rng = np.random.default_rng(881)
    eta = offset + x @ np.array([0.2, -0.15, 0.1])
    eta += z @ rng.normal(scale=0.15, size=z.shape[1])
    y = (
        rng.poisson(np.exp(eta)) if family == "poisson" else rng.binomial(1, special.expit(eta))
    ).astype(float)

    def objective(parameters):
        linear = offset + design @ parameters
        cumulant = np.exp(linear) if family == "poisson" else np.logaddexp(0, linear)
        return np.dot(weights, cumulant - y * linear) + 0.5 * np.dot(parameters[p:], parameters[p:])

    def derivatives(parameters):
        linear = offset + design @ parameters
        mean = np.exp(linear) if family == "poisson" else special.expit(linear)
        working_weights = weights * (mean if family == "poisson" else mean * (1 - mean))
        gradient = design.T @ (weights * (mean - y))
        gradient[p:] += parameters[p:]
        hessian = design.T @ (working_weights[:, None] * design)
        hessian[p:, p:] += np.eye(z.shape[1])
        return gradient, hessian

    fit = optimize.minimize(
        objective,
        np.zeros(design.shape[1]),
        jac=lambda parameters: derivatives(parameters)[0],
        hess=lambda parameters: derivatives(parameters)[1],
        method="trust-exact",
        options={"gtol": 1e-10},
    )
    gradient, hessian = derivatives(fit.x)
    assert np.max(np.abs(gradient)) < 1e-5
    # Objective changes can reach roundoff before the gradient does. Refine the
    # dense reference with one Newton step and verify its stationary equation.
    fit.x -= linalg.solve(hessian, gradient, assume_a="pos")
    gradient, hessian = derivatives(fit.x)
    assert np.max(np.abs(gradient)) < 1e-9
    # The native objective uses response deviance, which differs from the
    # optimizer's canonical negative log likelihood by a response-only constant.
    linear = offset + design @ fit.x
    if family == "poisson":
        mean = np.exp(linear)
        response_deviance = 2 * np.dot(weights, special.xlogy(y, y / mean) - (y - mean))
    else:
        response_deviance = 2 * np.dot(weights, np.logaddexp(0, linear) - y * linear)
    expected = response_deviance + np.dot(fit.x[p:], fit.x[p:])
    expected += np.linalg.slogdet(hessian[p:, p:])[1]
    native_args = (y, *args[1:12], family, "log" if family == "poisson" else "logit")
    actual, beta, random, converged = _native_laplace(native_args)
    assert converged
    assert_allclose(beta, fit.x[:p], rtol=2e-7, atol=2e-8)
    assert_allclose(random, covariance @ fit.x[p:], rtol=2e-7, atol=2e-8)
    assert actual == pytest.approx(expected, rel=2e-8, abs=2e-7)
