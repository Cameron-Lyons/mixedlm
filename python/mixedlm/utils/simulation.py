from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mixedlm.estimation.reml import _build_lambda_blocks, _count_theta
from mixedlm.families.base import Family
from mixedlm.matrices.design import ModelMatrices, RandomEffectStructure


def _accepts_sampling_inputs(simulate: Callable[..., Any]) -> bool:
    """Whether a simulate hook takes the ``weights`` and ``trials`` keywords."""
    try:
        parameters = inspect.signature(simulate).parameters.values()
    except (TypeError, ValueError):
        return False
    names = {parameter.name for parameter in parameters}
    return {"weights", "trials"} <= names or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )


def simulate_glmm_response(
    family: Family,
    mu: NDArray[np.floating],
    weights: NDArray[np.floating],
    *,
    trials: NDArray[np.floating] | None = None,
    proportion_scale: bool = False,
    rng: Any | None = None,
) -> NDArray[np.floating]:
    """Draw conditional responses with the fitted trials and precision weights.

    Two-dimensional means hold one replicate per column. Grouped binomial
    draws are success counts for explicit denominators and proportions when
    the fitted numeric response used trial weights.
    """
    rng = np.random if rng is None else rng
    if mu.ndim > 1:
        weights = weights[:, None]
        trials = None if trials is None else trials[:, None]
    # Inspect the underlying function, which outlives the bound method.
    if _accepts_sampling_inputs(getattr(family.simulate, "__func__", family.simulate)):
        draws = family.simulate(mu, rng=rng, weights=weights, trials=trials)
        if proportion_scale and trials is not None:
            return draws / trials
        return draws
    # Legacy overrides supply their own draws on the fitted response scale.
    return family.simulate(mu, rng=rng)


def simulate_fitted_glmm_response(
    family: Family,
    mu: NDArray[np.floating],
    matrices: ModelMatrices,
    rng: Any | None = None,
) -> NDArray[np.floating]:
    """Draw on the original response scale, keeping trial counts separate from prior weights."""
    from mixedlm.families.binomial import Binomial, _binomial_trials

    trials = matrices.response_trials
    if isinstance(family, Binomial):
        trials = _binomial_trials(matrices.y, matrices.weights, trials)
    return simulate_glmm_response(
        family,
        mu,
        matrices.weights,
        trials=trials,
        proportion_scale=trials is not None and matrices.trials is None,
        rng=rng,
    )


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
