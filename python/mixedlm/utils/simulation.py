from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from mixedlm.estimation.reml import _build_lambda_blocks, _count_theta
from mixedlm.families.base import Family
from mixedlm.families.gamma import Gamma
from mixedlm.families.gaussian import Gaussian
from mixedlm.families.inverse_gaussian import InverseGaussian
from mixedlm.matrices.design import RandomEffectStructure


def simulate_glmm_response(
    family: Family,
    mu: NDArray[np.floating],
    weights: NDArray[np.floating],
    *,
    trials: NDArray[np.floating] | None = None,
    rng: Any | None = None,
) -> NDArray[np.floating]:
    """Draw conditional responses with the fitted trials and precision weights."""
    rng = np.random if rng is None else rng
    if family.__class__.__name__ == "Binomial" and trials is not None:
        mu = family.clamp_mu(mu, eps=1e-6)
        counts = trials.astype(np.int64)
        if mu.ndim > 1:
            counts = counts[:, None]
        return rng.binomial(counts, mu).astype(np.float64)

    precision = weights if mu.ndim == 1 else weights[:, None]
    # Both subclass and instance overrides must retain control of their draws.
    simulation_method = getattr(family.simulate, "__func__", None)
    if isinstance(family, Gaussian) and simulation_method is Gaussian.simulate:
        return rng.normal(mu, 1 / np.sqrt(precision))
    if isinstance(family, Gamma) and simulation_method is Gamma.simulate:
        mu = np.minimum(family.clamp_mu(mu, eps=1e-6), 1e10)
        return rng.gamma(precision, mu / precision)
    if isinstance(family, InverseGaussian) and simulation_method is InverseGaussian.simulate:
        mu = np.minimum(family.clamp_mu(mu, eps=1e-6), 1e10)
        return rng.wald(mu, precision)
    return family.simulate(mu, rng=rng)


def simulate_random_effects(
    theta: NDArray[np.floating],
    structures: list[RandomEffectStructure],
    sigma: float = 1.0,
    *,
    rng: Any | None = None,
) -> NDArray[np.float64]:
    """Draw one set of random effects using the fitted covariance factors."""
    expected = _count_theta(structures)
    if theta.ndim != 1 or theta.size != expected:
        raise ValueError(f"theta must be a one-dimensional array of exactly {expected} values")
    rng = np.random if rng is None else rng
    result = np.empty(sum(s.n_levels * s.n_terms for s in structures), dtype=np.float64)
    start = 0
    theta_start = 0
    for structure in structures:
        theta_stop = theta_start + _count_theta([structure])
        parameters = theta[theta_start:theta_stop]
        size = structure.n_levels * structure.n_terms
        standard = rng.standard_normal((structure.n_levels, structure.n_terms))
        if not structure.correlated and structure.cov_type == "us":
            # Independent effects need a scale vector, even for wide designs.
            # This also avoids dense matrix products in serial bootstrap draws.
            standard *= parameters
            result[start : start + size] = standard.ravel()
        else:
            factor = _build_lambda_blocks(parameters, [structure])[0]
            result[start : start + size] = (standard @ factor.T).ravel()
        start += size
        theta_start = theta_stop
    result *= sigma
    return result


def simulation_parameters(
    theta: NDArray[np.floating],
    structures: list[RandomEffectStructure],
) -> tuple[NDArray[np.floating], list[bool]]:
    """Encode structured factors for the native unstructured batch sampler."""
    correlated = [s.correlated or s.cov_type in ("cs", "ar1") for s in structures]
    if not any(s.cov_type in ("cs", "ar1") for s in structures):
        return theta, correlated

    blocks = _build_lambda_blocks(theta, structures)
    packed = [
        factor[np.tril_indices(s.n_terms)] if is_correlated else np.diag(factor)
        for s, factor, is_correlated in zip(structures, blocks, correlated, strict=True)
    ]
    return np.concatenate(packed), correlated
