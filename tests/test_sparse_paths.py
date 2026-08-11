from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from mixedlm.estimation.reml import LMMOptimizer
from mixedlm.formula.parser import parse_formula
from mixedlm.inference.ddf import kenward_roger_df
from mixedlm.matrices.design import build_model_matrices
from mixedlm.models.checks import ModelCheckError, check_nobs_vs_rankZ, run_model_checks
from mixedlm.models.control import GlmerControl
from mixedlm.models.lmer import lmer
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg


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


def test_adaptive_start_uses_sparse_random_design_without_densifying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _large_crossed_data()
    formula = parse_formula("y ~ x + (1 | group1) + (1 | group2)")
    matrices = build_model_matrices(formula, data)
    optimizer = LMMOptimizer(matrices, use_rust=False)
    beta, sigma = optimizer._fit_ols()
    residuals = optimizer._compute_ols_residuals(beta)

    def fail_toarray(self):
        raise AssertionError("adaptive start should not densify sparse random-design blocks")

    monkeypatch.setattr(sparse.csc_matrix, "toarray", fail_toarray)

    for structure in matrices.random_structures:
        theta = optimizer._get_adaptive_start_for_structure(structure, residuals, sigma)
        assert len(theta) == 1


def test_rank_check_detects_saturated_sparse_random_design_without_densifying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_obs = 500
    matrices = SimpleNamespace(
        y=np.zeros(n_obs),
        Z=sparse.eye(n_obs, format="csc"),
    )

    def fail_toarray(self, *args, **kwargs):
        raise AssertionError("rank validation should not densify a sparse random design")

    monkeypatch.setattr(sparse.csc_matrix, "toarray", fail_toarray)

    with pytest.raises(ModelCheckError, match=r"500\) <= rank\(Z\) \(500"):
        check_nobs_vs_rankZ(matrices, "stop")


def test_rank_check_allows_numerically_deficient_sparse_design() -> None:
    Z = sparse.csc_matrix(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    matrices = SimpleNamespace(y=np.zeros(Z.shape[0]), Z=Z)

    assert sparse.csgraph.structural_rank(Z) == Z.shape[0]
    check_nobs_vs_rankZ(matrices, "stop")


def test_generalized_rank_check_skips_equality_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_obs = 100
    matrices = SimpleNamespace(
        y=np.zeros(n_obs),
        X=np.ones((n_obs, 1)),
        Z=sparse.eye(n_obs, format="csc"),
        random_structures=[],
    )

    def fail_svds(*args, **kwargs):
        raise AssertionError("equality is allowed, so rank computation is unnecessary")

    monkeypatch.setattr(sparse_linalg, "svds", fail_svds)

    control = GlmerControl(
        check_nobs_vs_rankZ="stop",
        check_nlev_gtr_1="ignore",
        check_nlev_gtreq_5="ignore",
        check_nobs_vs_nlev="ignore",
        check_rankX="ignore",
        check_scaleX="ignore",
    )
    run_model_checks(matrices, control)


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
