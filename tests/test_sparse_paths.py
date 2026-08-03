from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from mixedlm.estimation.reml import LMMOptimizer
from mixedlm.formula.parser import parse_formula
from mixedlm.inference.ddf import kenward_roger_df
from mixedlm.matrices.design import build_model_matrices
from mixedlm.models.lmer import lmer
from scipy import sparse


def _large_crossed_data(
    n_obs: int = 1_000,
    n_group1: int = 200,
    n_group2: int = 125,
) -> pd.DataFrame:
    rng = np.random.default_rng(20240502)
    group1_idx = np.arange(n_obs) % n_group1
    group2_idx = (np.arange(n_obs) * 7) % n_group2
    x = rng.normal(size=n_obs)
    group1_effects = rng.normal(scale=0.8, size=n_group1)
    group2_effects = rng.normal(scale=0.5, size=n_group2)
    y = 1.0 + 0.25 * x + group1_effects[group1_idx] + group2_effects[group2_idx]
    y += rng.normal(scale=0.2, size=n_obs)
    return pd.DataFrame(
        {
            "y": y,
            "x": x,
            "group1": [f"g1_{i}" for i in group1_idx],
            "group2": [f"g2_{i}" for i in group2_idx],
        }
    )


def test_adaptive_start_uses_retained_group_codes_without_sparse_slicing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _large_crossed_data()
    formula = parse_formula("y ~ x + (1 | group1) + (1 | group2)")
    matrices = build_model_matrices(formula, data)
    optimizer = LMMOptimizer(matrices, use_rust=False)
    beta, sigma = optimizer._fit_ols()
    residuals = optimizer._compute_ols_residuals(beta)

    for structure in matrices.random_structures:
        assert structure.level_indices is not None
        assert len(structure.level_indices) == len(data)

    def fail_getitem(self, key):
        raise AssertionError("adaptive start should not slice sparse random-design blocks")

    monkeypatch.setattr(sparse.csc_matrix, "__getitem__", fail_getitem)

    for structure in matrices.random_structures:
        theta = optimizer._get_adaptive_start_for_structure(structure, residuals, sigma)
        assert len(theta) == 1


def test_adaptive_start_includes_zero_slope_observations() -> None:
    data = pd.DataFrame(
        {
            "y": np.zeros(6),
            "x": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "group": ["a", "a", "b", "b", "c", "c"],
        }
    )
    formula = parse_formula("y ~ 1 + (0 + x | group)")
    matrices = build_model_matrices(formula, data)
    optimizer = LMMOptimizer(matrices, use_rust=False)
    structure = matrices.random_structures[0]
    residuals = np.array([0.2, 0.4, 0.4, 0.8, 0.2, 0.0])

    theta = optimizer._get_adaptive_start_for_structure(structure, residuals, sigma_ols=1.0)

    assert structure.level_indices is not None
    assert structure.level_indices.tolist() == [0, 0, 1, 1, 2, 2]
    expected_group_means = np.array([0.3, 0.6, 0.1])
    expected = np.sqrt(np.var(expected_group_means, ddof=1))
    assert theta == pytest.approx([expected])


def test_adaptive_start_falls_back_without_retained_group_codes() -> None:
    data = _large_crossed_data(n_obs=120, n_group1=12, n_group2=10)
    formula = parse_formula("y ~ x + (1 | group1)")
    matrices = build_model_matrices(formula, data)
    optimizer = LMMOptimizer(matrices, use_rust=False)
    beta, sigma = optimizer._fit_ols()
    residuals = optimizer._compute_ols_residuals(beta)
    expected = optimizer._get_adaptive_start_for_structure(
        matrices.random_structures[0], residuals, sigma
    )

    structure = replace(matrices.random_structures[0], level_indices=None)
    fallback_matrices = replace(matrices, random_structures=[structure])
    fallback_optimizer = LMMOptimizer(fallback_matrices, use_rust=False)

    actual = fallback_optimizer._get_adaptive_start_for_structure(structure, residuals, sigma)

    assert actual == pytest.approx(expected)


def test_kenward_roger_keeps_large_observation_covariance_sparse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _large_crossed_data(n_obs=160, n_group1=40, n_group2=32)
    model = lmer("y ~ x + (1 | group1) + (1 | group2)", data, REML=True)
    original_toarray = sparse.csc_matrix.toarray

    def reject_large_observation_covariance(self, *args, **kwargs):
        if self.shape[0] == model.matrices.n_obs and self.shape[1] == model.matrices.n_obs:
            raise AssertionError(
                "Kenward-Roger should not densify the n x n observation covariance"
            )
        return original_toarray(self, *args, **kwargs)

    monkeypatch.setattr(sparse.csc_matrix, "toarray", reject_large_observation_covariance)

    result = kenward_roger_df(model)
    assert result.df.shape == (model.matrices.n_fixed,)
    assert np.all(np.isfinite(result.df))
