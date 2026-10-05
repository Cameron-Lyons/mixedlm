"""LMM test problems and dense likelihood oracles independent of the profiled solver."""

import math
from dataclasses import replace
from decimal import Decimal, localcontext

import numpy as np
import pandas as pd
from mixedlm.estimation.reml import _build_lambda
from mixedlm.formula.parser import parse_formula, set_cov_type
from mixedlm.matrices.design import ModelMatrices, RandomEffectStructure, build_model_matrices
from numpy.testing import assert_array_equal
from scipy import linalg, optimize, sparse

from tests._glmm_oracles import mode_problem


def data_fixture():
    """Unbalanced groups with weights, an offset and random intercepts and slopes."""
    rng = np.random.default_rng(672)
    group = np.repeat(np.arange(7), [4, 6, 5, 7, 3, 8, 5])
    x = rng.normal(size=len(group)) + 0.15 * group
    z = rng.normal(size=len(group))
    offset = 0.2 * np.cos(np.arange(len(group)))
    weights = np.geomspace(0.5, 2.0, len(group))
    effects = rng.normal(scale=[0.8, 0.3, 0.25], size=(7, 3))
    y = 1.1 + 0.5 * x - 0.3 * z + offset
    y += effects[group, 0] + effects[group, 1] * x + effects[group, 2] * z
    y += rng.normal(scale=0.4 / np.sqrt(weights))
    return pd.DataFrame(dict(y=y, x=x, z=z, g=group)), weights, offset


def matrices_fixture(kind="intercept"):
    """Model matrices for one of several fixed and random-effect layouts of data_fixture()."""
    data, weights, offset = data_fixture()
    data["h"] = np.arange(len(data)) % 5
    formula = {
        "fixed": "y ~ x + z",
        "no_fixed": "y ~ 0 + (1 | g)",
        "intercept": "y ~ x + z + (1 | g)",
        "correlated": "y ~ x + z + (x | g)",
        "slope": "y ~ x + z + (0 + x | g)",
        "crossed": "y ~ x + z + (x | g) + (1 | h)",
        "cs": "y ~ x + z + (x + z | g)",
        "ar1": "y ~ x + z + (x + z | g)",
    }[kind]
    parsed = set_cov_type(formula, kind) if kind in {"cs", "ar1"} else parse_formula(formula)
    return build_model_matrices(parsed, data, weights=weights, offset=offset)


def native_arguments(matrices):
    """Keyword arguments of the native LmmDesign for these matrices."""
    z = matrices.Z.tocsc()
    return dict(
        x=matrices.X.copy(),
        z_data=z.data.copy(),
        z_indices=z.indices.astype(np.int64),
        z_indptr=z.indptr.astype(np.int64),
        z_shape=z.shape,
        weights=matrices.weights.copy(),
        offset=matrices.offset.copy(),
        n_levels=[s.n_levels for s in matrices.random_structures],
        n_terms=[s.n_terms for s in matrices.random_structures],
        correlated=[s.correlated for s in matrices.random_structures],
    )


def parameters(matrices):
    """A regular interior theta for every random-effect structure."""
    values = []
    for structure in matrices.random_structures:
        if structure.cov_type in {"cs", "ar1"}:
            values.extend([0.8, 0.2])
        elif structure.correlated:
            values.extend(
                0.8 if i == j else 0.1 for i in range(structure.n_terms) for j in range(i + 1)
            )
        else:
            values.extend([0.8] * structure.n_terms)
    return np.array(values)


def linear_data(signal=0.7):
    """Six balanced groups whose between-group signal sets the variance optimum."""
    return pd.DataFrame(
        {
            "y": np.tile([-1.0, -1.0, 1.0, 1.0], 6) + np.repeat(np.tile([-signal, signal], 3), 4),
            "g": np.repeat(np.arange(6), 4),
        }
    )


def wide_problem(width, independent, singular, fixed=True):
    """One structure of `width` overlapping terms; returns matrices, theta and dense factor."""
    rng = np.random.default_rng(524)
    n, levels = 96, 2
    z = rng.normal(scale=0.2, size=(n, width * levels))
    # Include cross-level entries so a transform cannot silently omit overlap.
    z[rng.random(z.shape) < 0.4] = 0.0
    x = np.column_stack((np.ones(n), rng.normal(size=(n, 2)))) if fixed else np.empty((n, 0))
    lower = np.diag(np.linspace(0.3, 0.8, width))
    if not independent:
        lower[np.tril_indices(width, -1)] = rng.uniform(-0.03, 0.03, width * (width - 1) // 2)
    if singular:
        lower[:, -1] = 0
    structure = RandomEffectStructure(
        "group", [f"x{i}" for i in range(width)], levels, width, not independent, {"a": 0, "b": 1}
    )
    theta = lower.diagonal().copy() if independent else lower[np.tril_indices(width)]
    matrices = ModelMatrices(
        y=rng.normal(size=n),
        X=x,
        Z=sparse.csc_matrix(z),
        fixed_names=["Intercept", "x", "z"] if fixed else [],
        random_structures=[structure],
        n_obs=n,
        n_fixed=x.shape[1],
        n_random=z.shape[1],
        weights=np.geomspace(0.4, 2.0, n),
        offset=np.linspace(-0.2, 0.3, n),
    )
    return matrices, theta, linalg.block_diag(*[lower] * levels)


def separate_structures(widths, independent, variance, fixed):
    """wide_problem() and a second structure, each on its own half of the rows."""
    left_width, right_width = widths
    matrices, theta, _ = wide_problem(left_width, independent, variance == "singular", fixed)
    rows = np.arange(matrices.n_obs)
    left = matrices.Z.toarray()
    left *= (rows[:, None] < matrices.n_obs // 2) & (
        rows[:, None] % 2 == np.arange(left.shape[1]) // left_width
    )
    right = np.random.default_rng(710).normal(scale=0.15, size=(matrices.n_obs, 3 * right_width))
    right *= (rows[:, None] >= matrices.n_obs // 2) & (
        rows[:, None] % 3 == np.arange(right.shape[1]) // right_width
    )
    lower = np.diag(np.linspace(0.3, 0.7, right_width))
    if independent:
        lower[np.tril_indices(right_width, -1)] = -0.02
    if variance == "singular":
        lower[:, -1] = 0
    theta = np.concatenate(
        (theta, lower[np.tril_indices(right_width)] if independent else lower.diagonal())
    )
    if variance == "zero":
        theta[:] = 0
    other = RandomEffectStructure(
        "other",
        [f"z{i}" for i in range(right_width)],
        3,
        right_width,
        independent,
        {"a": 0, "b": 1, "c": 2},
    )
    design = sparse.csc_matrix(np.column_stack((left, right)))
    matrices = replace(
        matrices,
        Z=design,
        n_random=design.shape[1],
        random_structures=[*matrices.random_structures, other],
    )
    return matrices, theta


def fixed_effect_problem(widths, fixed, coupled, diagonal, variance):
    """separate_structures() with `fixed` normal fixed-effect columns, the first an intercept."""
    matrices, theta = separate_structures(widths, diagonal, variance, False)
    x = np.random.default_rng(714).normal(size=(matrices.n_obs, fixed))
    if fixed:
        x[:, 0] = 1
    z = matrices.Z
    if coupled:
        columns = np.roll(np.arange(matrices.n_random), widths[0])
        z = (z + 0.15 * z[:, columns]).tocsc()
    return replace(
        matrices,
        X=x,
        Z=z,
        n_fixed=fixed,
        fixed_names=[f"x{i}" for i in range(fixed)],
    ), theta


def large_intercepts(levels, fixed):
    """Three rows per level of one scalar intercept term."""
    n = 3 * levels
    rows = np.arange(n)
    groups = rows % levels
    offset = 0.1 * np.cos(rows)
    matrices = ModelMatrices(
        y=offset + 0.3 + 0.4 * np.sin(groups) + 0.1 * np.sin(rows * 0.137),
        X=np.ones((n, 1)) if fixed else np.empty((n, 0)),
        Z=sparse.csc_matrix((np.ones(n), (rows, groups)), shape=(n, levels)),
        fixed_names=["Intercept"] if fixed else [],
        random_structures=[RandomEffectStructure("g", ["Intercept"], levels, 1, True, {})],
        n_obs=n,
        n_fixed=int(fixed),
        n_random=levels,
        weights=np.geomspace(0.5, 2.0, n),
        offset=offset,
    )
    return matrices, groups


def large_slopes(levels, width):
    """large_intercepts() with `width` correlated terms per level."""
    matrices, groups = large_intercepts(levels, True)
    terms = np.random.default_rng(714).normal(scale=0.3, size=(matrices.n_obs, width))
    terms[:, 0] = 1.0
    columns = groups[:, None] * width + np.arange(width)
    z = sparse.csc_matrix(
        (terms.ravel(), (np.repeat(np.arange(matrices.n_obs), width), columns.ravel())),
        shape=(matrices.n_obs, levels * width),
    )
    structure = replace(
        matrices.random_structures[0], n_terms=width, term_names=[f"x{i}" for i in range(width)]
    )
    matrices = replace(matrices, Z=z, n_random=z.shape[1], random_structures=[structure])
    lower = np.diag(np.linspace(0.3, 0.7, width))
    lower[np.tril_indices(width, -1)] = 0.03
    return matrices, terms, lower[np.tril_indices(width)]


def dominant_random_effects(fixed, response_scale=1.0):
    """Group effects six orders of magnitude above the residual scale."""
    matrices, _, _ = mode_problem("gaussian", "mode_only", n_obs=64, n_groups=4)
    row = np.arange(matrices.n_obs)
    groups = row % 4
    x = np.sin(row * 0.17)
    if fixed:
        # Keep the fixed column distinct from group intercepts even at large theta.
        means = np.bincount(groups, weights=matrices.weights * x) / np.bincount(
            groups, weights=matrices.weights
        )
        x -= means[groups]
        matrices = replace(matrices, X=x[:, None], n_fixed=1, fixed_names=["x"])
    fixed_part = 0.3 * x if fixed else 0.0
    response = response_scale * (1e6 * np.sin(groups) + fixed_part + 1e-3 * np.cos(row))
    return replace(matrices, y=response + matrices.offset), groups


def residual_problem(layout, size, pattern):
    """mode_problem() designs with overlapping columns or rows without random effects."""
    matrices, _, _ = mode_problem("gaussian", layout, n_obs=size, n_groups=16)
    if pattern == "overlap":
        columns = np.roll(np.arange(matrices.n_random), 2)
        matrices = replace(matrices, Z=(matrices.Z + 0.15 * matrices.Z[:, columns]).tocsc())
    elif pattern == "empty_rows":
        mask = (np.arange(size) % 9 != 0).astype(float)
        matrices = replace(matrices, Z=(sparse.diags(mask) @ matrices.Z).tocsc())
    return matrices


def observation_likelihood(matrices, theta, reml):
    """Dense observation-space oracle, independent of the profiled solver."""
    covariance = np.diag(1 / matrices.weights)
    position = 0
    blocks = []
    for structure in matrices.random_structures:
        width = structure.n_terms
        if structure.cov_type in {"cs", "ar1"}:
            scale, rho = theta[position : position + 2]
            position += 2
            correlation = (
                (1 - rho) * np.eye(width) + rho
                if structure.cov_type == "cs"
                else rho ** np.abs(np.arange(width)[:, None] - np.arange(width))
            )
            block = scale**2 * correlation
        else:
            count = width * (width + 1) // 2 if structure.correlated else width
            factor = np.zeros((width, width))
            if structure.correlated:
                factor[np.tril_indices(width)] = theta[position : position + count]
            else:
                np.fill_diagonal(factor, theta[position : position + count])
            position += count
            block = factor @ factor.T
        blocks.extend([block] * structure.n_levels)
    if blocks:
        z = matrices.Z.toarray()
        covariance += z @ linalg.block_diag(*blocks) @ z.T
    inverse_x = np.linalg.solve(covariance, matrices.X)
    information = matrices.X.T @ inverse_x
    y = matrices.y - matrices.offset
    beta = np.linalg.solve(information, inverse_x.T @ y)
    residual = y - matrices.X @ beta
    rss = residual @ np.linalg.solve(covariance, residual)
    df = matrices.n_obs - matrices.n_fixed if reml else matrices.n_obs
    return (
        df * (1 + np.log(2 * np.pi * rss / df))
        + np.linalg.slogdet(covariance)[1]
        + (np.linalg.slogdet(information)[1] if reml else 0)
    )


def observation_gradient(matrices, theta, reml):
    """Differentiate the observation covariance, including every grouping factor."""
    z = matrices.Z.toarray()
    transformed = z @ _build_lambda(theta, matrices.random_structures).toarray()
    covariance = np.diag(1 / matrices.weights) + transformed @ transformed.T
    precision = linalg.cho_solve(linalg.cho_factor(covariance), np.eye(matrices.n_obs))
    x, y = matrices.X, matrices.y - matrices.offset
    projector = precision
    if matrices.n_fixed:
        information = x.T @ precision @ x
        beta = linalg.solve(information, x.T @ precision @ y, assume_a="pos")
        residual = y - x @ beta
        if reml:
            projector = precision - precision @ x @ linalg.solve(information, x.T @ precision)
    else:
        residual = y
    projected = precision @ residual
    pwrss = residual @ projected
    df = matrices.n_obs - (matrices.n_fixed if reml else 0)
    score = projector - df / pwrss * np.outer(projected, projected)
    factor_score = 2 * z.T @ score @ transformed
    gradient, offset = [], 0
    for structure in matrices.random_structures:
        width = structure.n_terms
        positions = (
            [(i, j) for i in range(width) for j in range(i + 1)]
            if structure.correlated
            else [(i, i) for i in range(width)]
        )
        gradient.extend(
            sum(
                factor_score[offset + level * width + i, offset + level * width + j]
                for level in range(structure.n_levels)
            )
            for i, j in positions
        )
        offset += structure.n_levels * width
    return np.array(gradient)


def direct_profiled_likelihood(
    theta: np.ndarray,
    matrices: ModelMatrices,
    reml: bool,
) -> dict[str, np.ndarray | float]:
    lambda_matrix = _build_lambda(theta, matrices.random_structures).toarray()
    z_dense = matrices.Z.toarray()
    z_lambda = z_dense @ lambda_matrix
    sqrt_w = np.sqrt(matrices.weights)
    weighted_z_lambda = sqrt_w[:, None] * z_lambda
    weighted_x = sqrt_w[:, None] * matrices.X
    weighted_y = sqrt_w * (matrices.y - matrices.offset)

    marginal_cov = np.eye(matrices.n_obs) + weighted_z_lambda @ weighted_z_lambda.T
    chol_cov = linalg.cho_factor(marginal_cov, lower=True)
    cov_inv_x = linalg.cho_solve(chol_cov, weighted_x)
    cov_inv_y = linalg.cho_solve(chol_cov, weighted_y)
    information = weighted_x.T @ cov_inv_x
    beta = linalg.solve(information, weighted_x.T @ cov_inv_y, assume_a="pos")

    weighted_resid = weighted_y - weighted_x @ beta
    pwrss = float(weighted_resid @ linalg.cho_solve(chol_cov, weighted_resid))
    denom = matrices.n_obs - matrices.n_fixed if reml else matrices.n_obs
    sigma2 = pwrss / denom
    logdet_cov = float(np.linalg.slogdet(marginal_cov)[1])
    deviance = (
        denom * (1.0 + np.log(2.0 * np.pi * sigma2))
        + logdet_cov
        - float(np.sum(np.log(matrices.weights)))
    )
    if reml:
        deviance += float(np.linalg.slogdet(information)[1])

    marginal_resid = matrices.y - matrices.offset - matrices.X @ beta
    system = (
        np.eye(matrices.n_random)
        + lambda_matrix.T @ (z_dense.T @ (matrices.weights[:, None] * z_dense)) @ lambda_matrix
    )
    rhs = lambda_matrix.T @ (z_dense.T @ (matrices.weights * marginal_resid))
    spherical_effects = linalg.solve(system, rhs, assume_a="pos")
    random_effects = lambda_matrix @ spherical_effects
    conditional_resid = marginal_resid - z_dense @ random_effects
    wrss = float(np.dot(matrices.weights * conditional_resid, conditional_resid))
    ussq = float(np.dot(spherical_effects, spherical_effects))

    return {
        "deviance": float(deviance),
        "beta": beta,
        "sigma": float(np.sqrt(sigma2)),
        "u": random_effects,
        "wrss": wrss,
        "ussq": ussq,
        "pwrss": pwrss,
        "fixed_information": information,
    }


def groupwise_likelihood(matrices, groups, theta, reml):
    """High-precision within/between-group decomposition, without normal matrices."""
    with localcontext() as context:
        context.prec = 60
        y = [Decimal.from_float(float(v)) for v in matrices.y - matrices.offset]
        weights = [Decimal.from_float(float(v)) for v in matrices.weights]
        x = (
            [Decimal.from_float(float(v)) for v in matrices.X[:, 0]]
            if matrices.n_fixed
            else [Decimal(0)] * matrices.n_obs
        )
        variance = Decimal.from_float(float(theta)) ** 2
        blocks = []
        information, rhs = Decimal(0), Decimal(0)
        for group in np.unique(groups):
            indices = np.flatnonzero(groups == group)
            total = sum(weights[i] for i in indices)
            x_mean = sum(weights[i] * x[i] for i in indices) / total
            y_mean = sum(weights[i] * y[i] for i in indices) / total
            precision = 1 + variance * total
            information += sum(weights[i] * (x[i] - x_mean) ** 2 for i in indices)
            information += total * x_mean**2 / precision
            rhs += sum(weights[i] * (x[i] - x_mean) * (y[i] - y_mean) for i in indices)
            rhs += total * x_mean * y_mean / precision
            blocks.append((indices, total, x_mean, y_mean, precision))
        beta = rhs / information if matrices.n_fixed else Decimal(0)
        pwrss = Decimal(0)
        for indices, total, x_mean, y_mean, precision in blocks:
            mean = y_mean - x_mean * beta
            pwrss += sum(weights[i] * (y[i] - x[i] * beta - mean) ** 2 for i in indices)
            pwrss += total * mean**2 / precision
        df = matrices.n_obs - matrices.n_fixed if reml else matrices.n_obs
        deviance = df * (1 + math.log(2 * math.pi * float(pwrss) / df))
        deviance += math.fsum(math.log(float(block[-1])) for block in blocks)
        deviance -= math.fsum(math.log(float(weight)) for weight in weights)
        if reml and matrices.n_fixed:
            deviance += math.log(float(information))
        return deviance, math.sqrt(float(pwrss) / df)


def decimal_mode_likelihood(matrices, theta):
    """Small, high-precision penalized solve that also permits singular factors."""
    with localcontext() as context:
        context.prec = 60

        def decimal_array(values):
            return np.vectorize(lambda value: Decimal.from_float(float(value)))(values)

        z = decimal_array(matrices.Z.toarray())
        factor = decimal_array(_build_lambda(theta, matrices.random_structures).toarray())
        weights = decimal_array(matrices.weights)
        y = decimal_array(matrices.y - matrices.offset)
        design = z @ factor
        size = matrices.n_random
        identity = decimal_array(np.eye(size))
        precision = design.T @ (weights[:, None] * design) + identity
        # LDL decomposition uses only Decimal arithmetic, including the solve.
        lower = identity.copy()
        diagonal = []
        for i in range(size):
            diagonal.append(precision[i, i] - sum(lower[i, k] ** 2 * diagonal[k] for k in range(i)))
            for j in range(i + 1, size):
                lower[j, i] = (
                    precision[j, i] - sum(lower[j, k] * lower[i, k] * diagonal[k] for k in range(i))
                ) / diagonal[i]

        def solve(rhs):
            forward = []
            for i in range(size):
                forward.append(rhs[i] - sum(lower[i, k] * forward[k] for k in range(i)))
            result = [value / scale for value, scale in zip(forward, diagonal, strict=True)]
            for i in reversed(range(size)):
                result[i] -= sum(lower[k, i] * result[k] for k in range(i + 1, size))
            return np.array(result)

        spherical = solve(design.T @ (weights * y))
        residual = y - design @ spherical
        pwrss = sum(weights * residual**2) + sum(spherical**2)
        logdet = sum(math.log(float(value)) for value in diagonal)
        deviance = matrices.n_obs * (1 + math.log(2 * math.pi * float(pwrss) / matrices.n_obs))
        deviance += logdet - np.log(matrices.weights).sum()
        inverse = np.column_stack([solve(column) for column in identity])
        gradient = []
        for parameter in range(len(theta)):
            basis = np.zeros_like(theta)
            basis[parameter] = 1
            derivative = decimal_array(_build_lambda(basis, matrices.random_structures).toarray())
            d_design = z @ derivative
            d_precision = d_design.T @ (weights[:, None] * design)
            d_precision += design.T @ (weights[:, None] * d_design)
            d_logdet = np.trace(inverse @ d_precision)
            d_pwrss = -2 * sum(weights * residual * (d_design @ spherical))
            gradient.append(float(d_logdet + matrices.n_obs / pwrss * d_pwrss))
        return deviance, np.asarray(gradient)


def grouped_slope_oracle(matrices, terms, theta, reml):
    """Independent three-observation covariance systems, with no q-by-q array."""
    levels, width = matrices.random_structures[0].n_levels, terms.shape[1]
    design = terms.reshape(3, levels, width).transpose(1, 0, 2)
    lower = np.zeros((width, width))
    lower[np.tril_indices(width)] = theta
    transformed = design @ lower
    covariance = transformed @ transformed.transpose(0, 2, 1)
    diagonal = np.arange(3)
    covariance[:, diagonal, diagonal] += 1 / matrices.weights.reshape(3, levels).T
    precision = np.linalg.inv(covariance)
    y = (matrices.y - matrices.offset).reshape(3, levels).T
    information = precision.sum()
    beta = np.sum(np.einsum("gij,gj->gi", precision, y)) / information
    residual = y - beta
    projected = np.einsum("gij,gj->gi", precision, residual)
    pwrss = np.sum(residual * projected)
    df = matrices.n_obs - int(reml)
    sign, logdet = np.linalg.slogdet(covariance)
    assert_array_equal(sign, np.ones(levels))
    value = df * (1 + np.log(2 * np.pi * pwrss / df)) + logdet.sum()
    if reml:
        value += np.log(information)
    fixed_projection = precision.sum(axis=2)
    gradient = []
    for row, column in zip(*np.tril_indices(width), strict=True):
        basis = np.zeros((width, width))
        basis[row, column] = 1.0
        changed = design @ basis
        derivative = changed @ transformed.transpose(0, 2, 1)
        derivative += derivative.transpose(0, 2, 1).copy()
        score = np.einsum("gij,gji->", precision, derivative)
        score -= df / pwrss * np.einsum("gi,gij,gj->", projected, derivative, projected)
        if reml:
            score -= (
                np.einsum("gi,gij,gj->", fixed_projection, derivative, fixed_projection)
                / information
            )
        gradient.append(score)
    return value, gradient, beta, np.sqrt(pwrss / df)


def observation_space_reference(data, *, slopes):
    y = data["Reaction" if slopes else "diameter"].to_numpy(dtype=float)
    n = len(y)
    if slopes:
        days = data["Days"].to_numpy(dtype=float)
        X = np.column_stack((np.ones(n), days))
        membership = data["Subject"].to_numpy()
        same = (membership[:, None] == membership[None, :]).astype(float)

        def covariance(theta):
            lower = np.array([[theta[0], 0.0], [theta[1], theta[2]]])
            G = lower @ lower.T
            return np.eye(n) + same * (X @ G @ X.T)

        start = [1.0, 0.02, 0.2]
    else:
        X = np.ones((n, 1))
        grouping = [data[name].to_numpy() for name in ("plate", "sample")]
        same = [(values[:, None] == values[None, :]).astype(float) for values in grouping]

        def covariance(theta):
            return np.eye(n) + sum(
                value**2 * matrix for value, matrix in zip(theta, same, strict=True)
            )

        start = [1.5, 3.5]
    df = n - X.shape[1]

    def evaluate(theta):
        V = covariance(theta)
        factor = linalg.cho_factor(V, lower=True)
        inverse_X = linalg.cho_solve(factor, X)
        information = X.T @ inverse_X
        beta = np.linalg.solve(information, X.T @ linalg.cho_solve(factor, y))
        residuals = y - X @ beta
        inverse_residuals = linalg.cho_solve(factor, residuals)
        sigma_squared = float(residuals @ inverse_residuals / df)
        deviance = (
            2 * np.log(np.diag(factor[0])).sum()
            + np.linalg.slogdet(information)[1]
            + df * (1 + np.log(2 * np.pi * sigma_squared))
        )
        random_part = (V - np.eye(n)) @ inverse_residuals
        return {
            "deviance": float(deviance),
            "beta": beta,
            "sigma": np.sqrt(sigma_squared),
            "vcov": sigma_squared * np.linalg.inv(information),
            "fitted": X @ beta + random_part,
            "residuals": residuals - random_part,
        }

    optimum = optimize.minimize(
        lambda theta: evaluate(theta)["deviance"],
        start,
        method="Nelder-Mead",
        options={"xatol": 1e-9, "fatol": 1e-10, "maxiter": 2000},
    )
    assert optimum.success, optimum.message
    reference = evaluate(optimum.x)
    reference["theta"] = optimum.x
    return reference
