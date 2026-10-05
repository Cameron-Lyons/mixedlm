"""GLMM test problems and independently derived likelihood references."""

from dataclasses import replace

import numpy as np
import pandas as pd
from mixedlm import families
from mixedlm.estimation import laplace
from mixedlm.estimation.joint_glmm import JointGLMMObjective
from mixedlm.estimation.laplace import _native_glmm_args
from mixedlm.estimation.reml import _count_theta
from mixedlm.families import Poisson
from mixedlm.formula.parser import parse_formula
from mixedlm.matrices.design import build_model_matrices
from mixedlm.models.control import GlmerControl
from mixedlm.models.modular import GlmerParsedFormula, mkGlmerDevfun
from scipy import linalg, optimize, sparse, special


def mode_problem(kind, layout, *, n_obs=48, n_groups=4):
    """Weighted, offset GLMM matrices of the given family and layout with theta = 0.4."""
    rng = np.random.default_rng(3402)
    x = rng.uniform(-1, 1, n_obs)
    groups = np.arange(n_obs) % n_groups
    offset = 0.2 * np.cos(np.arange(n_obs))
    eta = 0.2 + 0.3 * x + 0.4 * np.sin(groups) + offset
    if kind == "binomial":
        trials = np.arange(n_obs) % 4 + 2
        y = rng.binomial(trials, special.expit(eta)) / trials
    elif kind == "poisson":
        y = rng.poisson(np.exp(eta))
    else:
        y = eta + rng.normal(scale=0.5, size=n_obs)
    data = pd.DataFrame(dict(y=y, x=x, g=groups, h=np.arange(n_obs) % 3))
    formulas = {
        "intercept": "y ~ x + (1 | g)",
        "slope": "y ~ x + (x | g)",
        "crossed": "y ~ x + (1 | g) + (1 | h)",
        "fixed_only": "y ~ x",
        "mode_only": "y ~ 0 + (1 | g)",
    }
    weights = np.geomspace(0.5, 2.0, n_obs)
    if kind == "binomial":
        weights *= trials
    matrices = build_model_matrices(
        parse_formula(formulas[layout]), data, weights=weights, offset=offset
    )
    family = {
        "gaussian": families.Gaussian,
        "binomial": families.Binomial,
        "poisson": families.Poisson,
    }[kind]()
    theta = np.full(_count_theta(matrices.random_structures), 0.4)
    return matrices, family, theta


NATIVE_GLMM_KEYS = (
    "y",
    "x",
    "z_data",
    "z_indices",
    "z_indptr",
    "z_shape",
    "weights",
    "offset",
    "theta",
    "n_levels",
    "n_terms",
    "correlated",
    "family",
    "link",
)


def native_glmm_arguments(kind="gaussian", layout="intercept"):
    """Keyword arguments of the native GLMM entry points for a mode_problem()."""
    matrices, family, theta = mode_problem(kind, layout)
    return dict(zip(NATIVE_GLMM_KEYS, _native_glmm_args(theta, matrices, family), strict=True))


def covariance_problem(layout, variance, weighted, family_name="gaussian", overlap=False):
    """Matrices, theta and dense block factor for covariance-transform layouts."""
    rng = np.random.default_rng(924)
    n = 120
    rows = rng.permutation(n)
    x, z, w = rng.uniform(-1, 1, (3, n))
    groups, other = rows % 6, rows % 5
    offset = 0.1 * np.sin(rows)
    eta = 0.3 + 0.2 * x - 0.1 * z + 0.15 * np.cos(groups) + offset
    if family_name == "poisson":
        y = rng.poisson(np.exp(eta)).astype(float)
    elif family_name == "binomial":
        y = rng.binomial(1, 1 / (1 + np.exp(-eta))).astype(float)
    else:
        y = eta + rng.normal(scale=0.3, size=n)
    data = pd.DataFrame(dict(y=y, x=x, z=z, w=w, g=groups, h=other))
    formulas = {
        "intercept": "y ~ x + z + (1 | g)",
        "correlated": "y ~ x + z + (x + z | g)",
        "diagonal": "y ~ x + z + (x + z || g)",
        "mixed": "y ~ x + z + (x + z | g) + (w || h)",
        "crossed_slopes": "y ~ x + z + (x | g) + (0 + z | h)",
        "no_fixed": "y ~ 0 + (x + z | g)",
        "fixed": "y ~ x + z",
    }
    matrices = build_model_matrices(
        parse_formula(formulas[layout]),
        data,
        weights=np.linspace(0.4, 2.0, n) if weighted else np.ones(n),
        offset=offset,
    )
    theta, blocks = [], []
    for structure in matrices.random_structures:
        width = structure.n_terms
        lower = np.diag(np.linspace(0.45, 0.85, width))
        if structure.correlated:
            for i in range(width):
                for j in range(i):
                    lower[i, j] = 0.1 * (i + 1) * (-1) ** j
        if variance == "singular":
            lower[:, -1] = 0.0
        elif variance == "zero":
            lower[:] = 0.0
        theta.extend(lower[np.tril_indices(width)] if structure.correlated else lower.diagonal())
        blocks.extend([lower] * structure.n_levels)
    factor = linalg.block_diag(*blocks) if blocks else np.zeros((0, 0))
    if overlap and matrices.n_random:
        # Advanced designs can overlap multiple levels. A block covariance
        # factor must preserve the off-diagonal entries of their crossproduct.
        extra = rng.normal(scale=0.05, size=matrices.Z.shape)
        extra[rng.random(extra.shape) < 0.9] = 0.0
        matrices = replace(matrices, Z=(matrices.Z + sparse.csc_matrix(extra)).tocsc())
    return matrices, np.asarray(theta), factor


def glmm_deviance_args(matrices, theta, family_name):
    """Positional arguments of _rust.glmm_deviance, assembled independently."""
    z = matrices.Z.tocsc()
    return (
        matrices.y,
        matrices.X,
        z.data,
        z.indices.astype(np.int64),
        z.indptr.astype(np.int64),
        z.shape,
        matrices.weights,
        matrices.offset,
        theta,
        [s.n_levels for s in matrices.random_structures],
        [s.n_terms for s in matrices.random_structures],
        [s.correlated for s in matrices.random_structures],
        family_name,
        {"gaussian": "identity", "poisson": "log", "binomial": "logit"}[family_name],
    )


class CustomPoisson(Poisson):
    """A serializable custom family that intentionally uses Python evaluation."""


def make_glmm_objective(
    target, kind="poisson", order=1, layout="intercept", maxiter=100, family=None
):
    """A GLMM optimizer, joint objective or modular devfun and its parameters."""
    matrices, default_family, theta = mode_problem(kind, layout)
    family = default_family if family is None else family
    if target == "optimizer":
        obj = laplace.GLMMOptimizer(
            matrices, family, nAGQ=order, pirls_maxiter=maxiter, pirls_tol=1e-10
        )
    elif target == "joint":
        obj = JointGLMMObjective(
            matrices, family, nAGQ=order, pirls_maxiter=maxiter, pirls_tol=1e-10
        )
    else:
        parsed = GlmerParsedFormula(parse_formula("y ~ x + (1 | g)"), matrices, family)
        obj = mkGlmerDevfun(
            parsed,
            nAGQ=order,
            control=GlmerControl(pirls_maxiter=maxiter, tolPwrss=1e-10, nAGQ0initStep=False),
        )
    parameters = np.r_[theta, np.full(matrices.n_fixed, 0.2)] if "joint" in target else theta
    return obj, parameters


def model_data(kind):
    """Poisson or grouped binomial data with eight random intercepts."""
    rng = np.random.default_rng(318)
    group = np.repeat(np.arange(8), 8)
    x = rng.normal(scale=0.4, size=len(group))
    offset = 0.15 * np.sin(np.arange(len(group)))
    weights = np.linspace(0.8, 1.3, len(group))
    eta = 0.4 + 0.5 * x + np.repeat(np.linspace(-1, 1, 8), 8) + offset
    if kind == "poisson":
        y = rng.poisson(np.exp(eta))
        data = pd.DataFrame(dict(y=y, x=x, g=group))
        formula = "y ~ x + (1 | g)"
        family = families.Poisson()
    else:
        trials = np.arange(len(group)) % 5 + 3
        successes = rng.binomial(trials, special.expit(eta))
        data = pd.DataFrame(dict(successes=successes, trials=trials, x=x, g=group))
        formula = "successes / trials ~ x + (1 | g)"
        family = families.Binomial()
    return formula, data, family, weights, offset, group


def independent_deviance(fitted, group, kind, order):
    """Deviance of (theta, *beta) by brute-force quadrature, or an exact scalar Laplace mode."""
    y, weights, offset, x = (
        fitted.matrices.y,
        fitted.matrices.weights,
        fitted.matrices.offset,
        fitted.matrices.X,
    )
    if kind == "poisson":
        constant = 2 * np.sum(weights * (special.xlogy(y, y) - y))
    else:
        constant = 2 * np.sum(weights * (special.xlogy(y, y) + special.xlogy(1 - y, 1 - y)))
    nodes, node_weights = np.polynomial.hermite.hermgauss(240)
    nodes = np.sqrt(2) * nodes
    log_weights = np.log(node_weights) - np.log(np.pi) / 2

    def objective(parameters):
        theta, beta = parameters[0], parameters[1:]
        total = constant
        for index in np.unique(group):
            keep = group == index
            eta = x[keep] @ beta + offset[keep]
            yy, ww = y[keep], weights[keep]
            if order > 1:
                linear = eta + theta * nodes[:, None]
                cumulant = np.exp(linear) if kind == "poisson" else np.logaddexp(0, linear)
                log_likelihood = np.sum(ww * (yy * linear - cumulant), axis=1)
                total -= 2 * special.logsumexp(log_weights + log_likelihood)
            else:
                inverse = np.exp if kind == "poisson" else special.expit
                mode = optimize.brentq(
                    lambda u, w=ww, y=yy, e=eta, inv=inverse: u
                    - theta * (w @ (y - inv(e + theta * u))),
                    -30,
                    30,
                )
                linear = eta + theta * mode
                mu = inverse(linear)
                variance = mu if kind == "poisson" else mu * (1 - mu)
                cumulant = mu if kind == "poisson" else np.logaddexp(0, linear)
                total += 2 * np.sum(ww * (cumulant - yy * linear))
                total += mode**2 + np.log1p(theta**2 * (ww @ variance))
        return total

    return objective
