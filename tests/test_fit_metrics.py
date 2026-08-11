from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from mixedlm import ICCResult, R2NakagawaResult, families, glmer, icc, lmer, r2_nakagawa
from mixedlm.estimation.nlmm import _build_psi_matrix
from mixedlm.families import Binomial, Poisson
from mixedlm.nlme import SSmicmen
from scipy import sparse


@dataclass
class _FakeLinearModel:
    beta: np.ndarray
    sigma: float
    matrices: SimpleNamespace
    structures: list[SimpleNamespace]
    covariances: list[np.ndarray]

    def isLMM(self) -> bool:
        return True

    def isGLMM(self) -> bool:
        return False

    def isNLMM(self) -> bool:
        return False

    def _iter_random_cov_blocks(self, scale: float = 1.0):
        for structure, covariance in zip(self.structures, self.covariances, strict=True):
            yield structure, covariance * scale


@dataclass
class _FakeGeneralizedModel:
    beta: np.ndarray
    u: np.ndarray
    family: object
    matrices: SimpleNamespace
    structures: list[SimpleNamespace]
    covariances: list[np.ndarray]

    def isLMM(self) -> bool:
        return False

    def isGLMM(self) -> bool:
        return True

    def isNLMM(self) -> bool:
        return False

    def _iter_random_cov_blocks(self, scale: float = 1.0):
        for structure, covariance in zip(self.structures, self.covariances, strict=True):
            yield structure, covariance * scale


@dataclass
class _FakeNonlinearModel:
    model: object
    phi: np.ndarray
    theta: np.ndarray
    sigma: float
    x: np.ndarray
    random_params: list[int]
    group_var: str = "subject"

    def isLMM(self) -> bool:
        return False

    def isGLMM(self) -> bool:
        return False

    def isNLMM(self) -> bool:
        return True


@pytest.fixture
def linear_model() -> _FakeLinearModel:
    x = np.arange(4, dtype=np.float64)
    X = np.column_stack([np.ones(4), x])
    Z = sparse.csc_matrix(
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
            ]
        )
    )
    structure = SimpleNamespace(
        grouping_factor="subject",
        n_levels=2,
        n_terms=2,
    )
    matrices = SimpleNamespace(X=X, Z=Z, offset=np.zeros(4), weights=np.ones(4))
    relative_covariance = np.array([[0.25, 0.05], [0.05, 0.125]])
    return _FakeLinearModel(
        beta=np.array([1.0, 2.0]),
        sigma=2.0,
        matrices=matrices,
        structures=[structure],
        covariances=[relative_covariance],
    )


def _as_glmm(linear_model: _FakeLinearModel, family: object) -> _FakeGeneralizedModel:
    return _FakeGeneralizedModel(
        beta=np.array([-1.0, 0.2]),
        u=np.zeros(linear_model.matrices.Z.shape[1]),
        family=family,
        matrices=linear_model.matrices,
        structures=linear_model.structures,
        covariances=linear_model.covariances,
    )


class TestNakagawaR2:
    def test_linear_random_slope_decomposition(self, linear_model) -> None:
        result = r2_nakagawa(linear_model)

        fixed = np.var(np.array([1.0, 3.0, 5.0, 7.0]), ddof=1)
        random = np.mean(np.array([1.0, 1.9, 3.8, 6.7]))
        residual = 4.0
        total = fixed + random + residual

        assert isinstance(result, R2NakagawaResult)
        assert result.variance_fixed == pytest.approx(fixed)
        assert result.variance_random == pytest.approx(random)
        assert result.variance_residual == residual
        assert result.random_by_group == {"subject": pytest.approx(random)}
        assert result.marginal == pytest.approx(fixed / total)
        assert result.conditional == pytest.approx((fixed + random) / total)
        assert result.approximation == "gaussian"

    def test_fixed_offset_contributes_to_explained_variance(self, linear_model) -> None:
        linear_model.beta = np.array([2.0, 0.0])
        linear_model.matrices.offset = np.array([0.0, 1.0, 2.0, 3.0])

        result = r2_nakagawa(linear_model)

        assert result.variance_fixed == pytest.approx(np.var([2.0, 3.0, 4.0, 5.0], ddof=1))

    def test_no_random_effects_makes_conditional_equal_marginal(self, linear_model) -> None:
        linear_model.matrices.Z = sparse.csc_matrix((4, 0))
        linear_model.structures = []
        linear_model.covariances = []

        result = r2_nakagawa(linear_model)

        assert result.conditional == pytest.approx(result.marginal)
        assert result.variance_random == 0.0
        assert result.random_by_group == {}

    def test_crossed_groups_partition_random_variance(self, linear_model) -> None:
        item_structure = SimpleNamespace(grouping_factor="item", n_levels=2, n_terms=1)
        item_design = sparse.csc_matrix(
            np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            )
        )
        linear_model.matrices.Z = sparse.hstack(
            [linear_model.matrices.Z, item_design], format="csc"
        )
        linear_model.structures.append(item_structure)
        linear_model.covariances.append(np.array([[0.5]]))

        result = r2_nakagawa(linear_model)

        assert result.random_by_group["subject"] == pytest.approx(3.35)
        assert result.random_by_group["item"] == pytest.approx(2.0)
        assert sum(result.random_by_group.values()) == pytest.approx(result.variance_random)

    def test_poisson_lognormal_residual_variance(self, linear_model) -> None:
        model = _as_glmm(linear_model, Poisson())

        result = r2_nakagawa(model)

        eta = model.matrices.X @ model.beta
        mu = np.exp(eta)
        expected = np.mean(np.log1p(1.0 / mu))
        assert result.variance_residual == pytest.approx(expected)
        assert result.approximation == "lognormal"

    def test_poisson_delta_residual_variance(self, linear_model) -> None:
        model = _as_glmm(linear_model, Poisson())

        result = r2_nakagawa(model, approximation="delta")

        mu = np.exp(model.matrices.X @ model.beta)
        assert result.variance_residual == pytest.approx(np.mean(1.0 / mu))
        assert result.approximation == "delta"

    def test_binomial_uses_link_theoretical_variance(self, linear_model) -> None:
        model = _as_glmm(linear_model, Binomial())

        result = r2_nakagawa(model)

        assert result.variance_residual == pytest.approx(np.pi**2 / 3.0)
        assert result.approximation == "theoretical"

    def test_nonlinear_delta_random_variance(self) -> None:
        nonlinear = SSmicmen()
        x = np.array([0.5, 1.0, 2.0, 4.0])
        phi = np.array([10.0, 2.0])
        theta = np.array([0.2, 0.05, 0.1])
        model = _FakeNonlinearModel(
            model=nonlinear,
            phi=phi,
            theta=theta,
            sigma=0.5,
            x=x,
            random_params=[0, 1],
        )

        result = r2_nakagawa(model)

        gradient = nonlinear.gradient(phi, x)
        covariance = _build_psi_matrix(theta, 2) * model.sigma**2
        expected_random = np.mean(np.sum((gradient @ covariance) * gradient, axis=1))
        expected_fixed = np.var(nonlinear.predict(phi, x), ddof=1)
        assert result.variance_fixed == pytest.approx(expected_fixed)
        assert result.variance_random == pytest.approx(expected_random)
        assert result.variance_residual == pytest.approx(0.25)
        assert result.approximation == "gaussian-delta"


class TestICC:
    def test_adjusted_and_unadjusted_icc(self, linear_model) -> None:
        result = icc(linear_model)

        fixed = np.var(np.array([1.0, 3.0, 5.0, 7.0]), ddof=1)
        random = np.mean(np.array([1.0, 1.9, 3.8, 6.7]))
        residual = 4.0
        assert isinstance(result, ICCResult)
        assert result.adjusted == pytest.approx(random / (random + residual))
        assert result.unadjusted == pytest.approx(random / (fixed + random + residual))
        assert result.by_group["subject"] == pytest.approx(result.adjusted)
        assert result.by_group_unadjusted["subject"] == pytest.approx(result.unadjusted)

    def test_no_random_effects_returns_zero(self, linear_model) -> None:
        linear_model.matrices.Z = sparse.csc_matrix((4, 0))
        linear_model.structures = []
        linear_model.covariances = []

        result = icc(linear_model)

        assert result.adjusted == 0.0
        assert result.unadjusted == 0.0
        assert result.by_group == {}


class TestFitMetricValidation:
    def test_rejects_unknown_approximation(self, linear_model) -> None:
        with pytest.raises(ValueError, match="Unknown approximation"):
            r2_nakagawa(linear_model, approximation="unknown")

    def test_rejects_theoretical_for_non_binomial(self, linear_model) -> None:
        model = _as_glmm(linear_model, Poisson())

        with pytest.raises(ValueError, match="only available for binomial"):
            r2_nakagawa(model, approximation="theoretical")

    def test_rejects_non_model(self) -> None:
        with pytest.raises(TypeError, match="fitted linear"):
            r2_nakagawa(object())

    def test_result_helpers(self, linear_model) -> None:
        r2 = r2_nakagawa(linear_model)
        correlation = icc(linear_model)

        assert r2.as_dict()["marginal"] == r2.marginal
        assert correlation.as_dict()["adjusted"] == correlation.adjusted
        assert "marginal=" in str(r2)
        assert "adjusted=" in str(correlation)


class TestFittedModelIntegration:
    def test_linear_result(self) -> None:
        rng = np.random.default_rng(510)
        groups = np.repeat(np.arange(8), 8)
        x = rng.normal(size=len(groups))
        group_effect = rng.normal(scale=0.8, size=8)
        y = 1.0 + 0.7 * x + group_effect[groups] + rng.normal(scale=0.4, size=len(groups))
        data = pd.DataFrame({"y": y, "x": x, "group": groups.astype(str)})
        model = lmer("y ~ x + (1 | group)", data)

        r2 = r2_nakagawa(model)
        correlation = icc(model)

        assert 0.0 <= r2.marginal <= r2.conditional <= 1.0
        assert 0.0 <= correlation.unadjusted <= correlation.adjusted <= 1.0

    def test_generalized_result(self) -> None:
        rng = np.random.default_rng(511)
        groups = np.repeat(np.arange(10), 10)
        x = rng.normal(size=len(groups))
        eta = -0.6 + 0.5 * x + rng.normal(scale=0.3, size=10)[groups]
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta)))
        data = pd.DataFrame({"y": y, "x": x, "group": groups.astype(str)})
        model = glmer("y ~ x + (1 | group)", data, family=families.Binomial())

        r2 = r2_nakagawa(model)
        correlation = icc(model)

        assert r2.approximation == "theoretical"
        assert 0.0 <= r2.marginal <= r2.conditional <= 1.0
        assert 0.0 <= correlation.unadjusted <= correlation.adjusted <= 1.0
