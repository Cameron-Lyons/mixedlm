from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mixedlm.estimation.reml import _build_lambda_blocks
from mixedlm.matrices.design import RandomEffectStructure


def simulate_random_effects(
    theta: NDArray[np.floating],
    structures: list[RandomEffectStructure],
    sigma: float = 1.0,
) -> NDArray[np.float64]:
    """Draw one set of random effects using the fitted covariance factors."""
    result = np.empty(sum(s.n_levels * s.n_terms for s in structures), dtype=np.float64)
    start = 0
    for structure, factor in zip(structures, _build_lambda_blocks(theta, structures), strict=True):
        size = structure.n_levels * structure.n_terms
        standard = np.random.randn(structure.n_levels, structure.n_terms)
        result[start : start + size] = (standard @ factor.T).ravel()
        start += size
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
