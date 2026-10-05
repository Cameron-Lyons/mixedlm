"""Variance diagnostics checked against the covariance of observed responses."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from mixedlm import icc, lmer, lmerControl, r2_nakagawa
from mixedlm.models import shared_utils
from scipy import linalg, sparse

from tests.test_fit_metrics import _FakeLinearModel, _FakeNonlinearModel


@pytest.mark.parametrize("use_rust", [False, True])
def test_weighted_crossed_fit_matches_observation_covariance(use_rust):
    rng = np.random.default_rng(72)
    subject = np.repeat(np.arange(8), 8)
    item = np.tile(np.arange(8), 8)
    x = rng.normal(size=64)
    precision = np.exp(rng.normal(scale=0.6, size=64))
    y = (
        1.2
        + 0.7 * x
        + rng.normal(scale=0.8, size=8)[subject]
        + rng.normal(scale=0.4, size=8)[item]
        + rng.normal(scale=0.5, size=64) / np.sqrt(precision)
    )
    data = pd.DataFrame({"y": y, "x": x, "subject": subject, "item": item})
    model = lmer(
        "y ~ x + (1 | subject) + (1 | item)",
        data,
        weights=precision,
        control=lmerControl(use_rust=use_rust),
    )
    assert model.converged

    # Independent observation covariance: V = Z G Z' + sigma² W^-1.
    blocks = [
        np.kron(np.eye(structure.n_levels), covariance)
        for structure, covariance in model._iter_random_cov_blocks(scale=model.sigma**2)
    ]
    design = model.matrices.Z.toarray()
    random_covariance = design @ linalg.block_diag(*blocks) @ design.T
    fixed = np.var(model.matrices.X @ model.beta, ddof=1)
    random = np.trace(random_covariance) / len(data)
    residual = np.trace(np.diag(model.sigma**2 / precision)) / len(data)
    total = fixed + random + residual

    result = r2_nakagawa(model)
    correlation = icc(model)
    assert result.variance_residual == pytest.approx(residual)
    assert result.variance_random == pytest.approx(random)
    assert result.marginal == pytest.approx(fixed / total)
    assert result.conditional == pytest.approx((fixed + random) / total)
    assert correlation.adjusted == pytest.approx(random / (random + residual))
    assert sum(correlation.by_group.values()) == pytest.approx(correlation.adjusted)

    # The same response distribution can be parameterized with a common
    # precision multiplier, compensated by sigma and the relative covariance.
    scaled = replace(
        model,
        sigma=model.sigma * np.sqrt(16.0),
        theta=model.theta / np.sqrt(16.0),
        matrices=replace(model.matrices, weights=16.0 * model.matrices.weights),
    )
    assert r2_nakagawa(scaled).as_dict() == pytest.approx(result.as_dict())
    assert icc(scaled).as_dict() == pytest.approx(correlation.as_dict())


def test_nonlinear_residual_component_honors_precision():
    from mixedlm.nlme import SSmicmen

    class WeightedModel(_FakeNonlinearModel):
        def weights(self, copy=True):
            return np.array([0.25, 1.0, 4.0, 16.0])

    model = WeightedModel(
        model=SSmicmen(),
        phi=np.array([10.0, 2.0]),
        theta=np.array([0.2]),
        sigma=2.0,
        x=np.array([0.5, 1.0, 2.0, 4.0]),
        random_params=[0],
    )
    expected = (16.0 + 4.0 + 1.0 + 0.25) / 4
    assert r2_nakagawa(model).variance_residual == pytest.approx(expected)
    assert icc(model).variance_residual == pytest.approx(expected)


def test_nonlinear_offsets_contribute_to_fixed_prediction_variance():
    from mixedlm.nlme import SSmicmen

    class OffsetModel(_FakeNonlinearModel):
        def offset(self, copy=True):
            # Known offsets exactly cancel the population Michaelis-Menten curve.
            return -np.array([2.0, 10.0 / 3.0, 5.0, 20.0 / 3.0])

    model = OffsetModel(
        model=SSmicmen(),
        phi=np.array([10.0, 2.0]),
        theta=np.array([0.2]),
        sigma=2.0,
        x=np.array([0.5, 1.0, 2.0, 4.0]),
        random_params=[0],
    )
    result = r2_nakagawa(model)
    assert result.variance_fixed == pytest.approx(0.0, abs=1e-25)
    assert result.marginal == pytest.approx(0.0, abs=1e-25)
    assert result.variance_random > 0


def test_large_finite_components_have_finite_ratios():
    model = _FakeLinearModel(
        beta=np.array([1.0]),
        sigma=1.0,
        matrices=SimpleNamespace(
            X=np.array([[0.0], [np.sqrt(2.0) * 1e154]]),
            Z=sparse.csc_matrix(np.ones((2, 1))),
            offset=np.zeros(2),
            weights=np.full(2, 1e-308),
        ),
        structures=[SimpleNamespace(grouping_factor="group", n_terms=1, n_levels=1)],
        covariances=[np.array([[1e308]])],
    )
    with np.errstate(over="raise", invalid="raise"):
        result = r2_nakagawa(model)
        correlation = icc(model)
    assert result.variance_fixed == pytest.approx(1e308)
    assert result.variance_random == pytest.approx(1e308)
    assert result.variance_residual == pytest.approx(1e308)
    assert result.marginal == pytest.approx(1 / 3)
    assert result.conditional == pytest.approx(2 / 3)
    assert correlation.adjusted == pytest.approx(1 / 2)
    assert correlation.unadjusted == pytest.approx(1 / 3)


def test_gaussian_mean_residual_is_finite_when_one_row_exceeds_float_range():
    model = _FakeLinearModel(
        beta=np.array([1.0]),
        sigma=1.0,
        matrices=SimpleNamespace(
            X=np.ones((2, 1)),
            Z=sparse.csc_matrix((2, 0)),
            offset=np.zeros(2),
            weights=np.array([5e-309, 1.0]),
        ),
        structures=[],
        covariances=[],
    )
    result = r2_nakagawa(model)
    assert result.variance_residual == pytest.approx(1e308)
    assert result.marginal == 0


def test_large_common_baseline_keeps_small_fixed_variation():
    values = 1e15 + np.array([0.0, 0.25, 0.5, 0.75])
    model = _FakeLinearModel(
        beta=np.array([1.0]),
        sigma=1.0,
        matrices=SimpleNamespace(
            X=values[:, None],
            Z=sparse.csc_matrix((4, 0)),
            offset=np.zeros(4),
            weights=np.ones(4),
        ),
        structures=[],
        covariances=[],
    )
    expected = (0.375**2 + 0.125**2 + 0.125**2 + 0.375**2) / 3
    assert r2_nakagawa(model).variance_fixed == pytest.approx(expected)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_predictions_are_not_silently_dropped(invalid):
    model = _FakeLinearModel(
        beta=np.array([1.0]),
        sigma=1.0,
        matrices=SimpleNamespace(
            X=np.array([[1.0], [invalid]]),
            Z=sparse.csc_matrix((2, 0)),
            offset=np.zeros(2),
            weights=np.ones(2),
        ),
        structures=[],
        covariances=[],
    )
    with pytest.raises(ValueError, match="Fixed predictions must be finite"):
        r2_nakagawa(model)


def test_random_slope_projection_uses_bounded_sparse_buffers(monkeypatch):
    rng = np.random.default_rng(6)
    n, q = 11, 5
    term_design = rng.normal(size=(n, q))
    levels = np.arange(n) % 3
    design = np.zeros((n, 3 * q))
    for row in range(n):
        design[row, levels[row] * q : (levels[row] + 1) * q] = term_design[row]
    factor = rng.normal(size=(q, q))
    covariance = factor @ factor.T
    model = _FakeLinearModel(
        beta=np.array([0.0]),
        sigma=1.0,
        matrices=SimpleNamespace(
            X=np.ones((n, 1)),
            Z=sparse.csc_matrix(design),
            offset=np.zeros(n),
            weights=np.ones(n),
        ),
        structures=[SimpleNamespace(grouping_factor="group", n_terms=q, n_levels=3)],
        covariances=[covariance],
    )
    expected = np.trace(design @ np.kron(np.eye(3), covariance) @ design.T) / n
    monkeypatch.setattr(shared_utils, "_MAX_QUADRATIC_FORM_ELEMENTS", 10)
    original = sparse.csr_matrix.__matmul__
    buffers = []

    def checked_product(self, other):
        buffers.append(self.shape)
        assert self.shape[0] * self.shape[1] <= 10
        return original(self, other)

    def reject_dense(*args, **kwargs):
        pytest.fail("Diagnostics must retain a sparse observation design")

    monkeypatch.setattr(sparse.csr_matrix, "__matmul__", checked_product)
    monkeypatch.setattr(sparse.csr_matrix, "toarray", reject_dense)
    assert r2_nakagawa(model).variance_random == pytest.approx(expected)
    assert len(buffers) == 6


@pytest.mark.parametrize("invalid", [0.0, -1.0, np.nan, np.inf])
def test_invalid_gaussian_precision_rejected(invalid):
    model = _FakeLinearModel(
        beta=np.array([1.0]),
        sigma=1.0,
        matrices=SimpleNamespace(
            X=np.ones((2, 1)),
            Z=sparse.csc_matrix((2, 0)),
            offset=np.zeros(2),
            weights=np.array([1.0, invalid]),
        ),
        structures=[],
        covariances=[],
    )
    with pytest.raises(ValueError, match="weights must be finite and positive"):
        r2_nakagawa(model)
