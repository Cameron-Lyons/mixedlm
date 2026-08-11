from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from mixedlm import families, glmer
from mixedlm.diagnostics import (
    DispersionResult,
    ZeroInflationResult,
    check_overdispersion,
    check_zero_inflation,
    check_zeroinflation,
)
from mixedlm.models.lmer import LmerResult
from numpy.testing import assert_allclose
from scipy import stats


@dataclass
class _Matrices:
    y: np.ndarray
    weights: np.ndarray


class _GeneralizedModel:
    def __init__(
        self,
        y: list[float],
        mu: list[float],
        family,
        *,
        weights: list[float] | None = None,
        n_parameters: int = 1,
    ) -> None:
        self._mu = np.asarray(mu, dtype=float)
        weight_values = np.ones(len(y)) if weights is None else weights
        self.matrices = _Matrices(
            np.asarray(y, dtype=float), np.asarray(weight_values, dtype=float)
        )
        self.family = family
        self._n_parameters = n_parameters

    def isGLMM(self) -> bool:
        return True

    def fitted(self, type: str = "response", na_expand: bool = False) -> np.ndarray:
        assert type == "response"
        assert not na_expand
        return self._mu

    def npar(self) -> int:
        return self._n_parameters


def test_weighted_pearson_dispersion_matches_direct_calculation() -> None:
    y = np.array([0.0, 1.0, 4.0, 2.0])
    mu = np.array([0.5, 1.2, 2.5, 3.0])
    weights = np.array([1.0, 2.0, 0.5, 3.0])
    model = _GeneralizedModel(
        y.tolist(), mu.tolist(), families.Poisson(), weights=weights.tolist(), n_parameters=2
    )

    result = check_overdispersion(model)
    expected_chi_square = float(np.sum(weights * np.square(y - mu) / mu))
    expected_p = 2 * min(
        stats.chi2.cdf(expected_chi_square, 2), stats.chi2.sf(expected_chi_square, 2)
    )

    assert isinstance(result, DispersionResult)
    assert result.pearson_chi_square == pytest.approx(expected_chi_square)
    assert result.residual_df == 2
    assert result.dispersion_ratio == pytest.approx(expected_chi_square / 2)
    assert result.p_value == pytest.approx(expected_p)
    assert result.ci_low < result.dispersion_ratio < result.ci_high
    assert result.as_dict()["status"] == result.status
    assert "Pearson dispersion check" in str(result)


@pytest.mark.parametrize(
    ("alternative", "expected"),
    [
        ("greater", lambda statistic, df: stats.chi2.sf(statistic, df)),
        ("less", lambda statistic, df: stats.chi2.cdf(statistic, df)),
    ],
)
def test_dispersion_alternatives(alternative: str, expected) -> None:
    model = _GeneralizedModel([0, 5, 8, 1], [1, 1, 1, 1], families.Poisson())
    result = check_overdispersion(model, alternative=alternative)
    assert result.p_value == pytest.approx(expected(result.pearson_chi_square, result.residual_df))


def test_dispersion_flags_over_and_underdispersion() -> None:
    over = _GeneralizedModel([0] * 19 + [50], [1] * 20, families.Poisson())
    under = _GeneralizedModel([1] * 20, [1] * 20, families.Poisson())

    over_result = check_overdispersion(over, alternative="greater")
    under_result = check_overdispersion(under, alternative="less")

    assert over_result.is_overdispersed
    assert over_result.status == "overdispersed"
    assert under_result.is_underdispersed
    assert under_result.status == "underdispersed"


def test_poisson_zero_check_matches_closed_form_moments() -> None:
    y = np.array([0, 0, 1, 2, 3], dtype=float)
    mu = np.array([0.2, 0.5, 1.0, 2.0, 4.0])
    model = _GeneralizedModel(y.tolist(), mu.tolist(), families.Poisson())

    result = check_zero_inflation(model, alternative="two-sided")
    probabilities = np.exp(-mu)

    assert isinstance(result, ZeroInflationResult)
    assert result.observed_zeros == 2
    assert result.expected_zeros == pytest.approx(np.sum(probabilities))
    assert result.variance == pytest.approx(np.sum(probabilities * (1 - probabilities)))
    assert result.observed_to_expected == pytest.approx(2 / np.sum(probabilities))
    assert 0 <= result.p_value <= 1
    assert result.as_dict()["observed_zeros"] == 2
    assert "Zero-inflation check" in str(result)


def test_negative_binomial_zero_probabilities() -> None:
    theta = 2.5
    mu = np.array([0.5, 1.0, 3.0, 5.0])
    model = _GeneralizedModel([0, 1, 0, 4], mu.tolist(), families.NegativeBinomial(theta))

    result = check_zero_inflation(model)
    expected = np.sum(np.power(theta / (theta + mu), theta))

    assert result.expected_zeros == pytest.approx(expected)


def test_grouped_binomial_zero_probabilities_use_trial_weights() -> None:
    mu = np.array([0.2, 0.5, 0.8])
    trials = np.array([5.0, 2.0, 10.0])
    model = _GeneralizedModel(
        [0.0, 0.5, 0.8], mu.tolist(), families.Binomial(), weights=trials.tolist()
    )

    result = check_zero_inflation(model)

    assert result.observed_zeros == 1
    assert result.expected_zeros == pytest.approx(np.sum(np.power(1 - mu, trials)))


def test_zero_check_alias() -> None:
    model = _GeneralizedModel([0, 1], [0.5, 1.5], families.Poisson())
    assert check_zeroinflation(model) == check_zero_inflation(model)


def test_zero_check_handles_deterministic_probabilities() -> None:
    model = _GeneralizedModel([0, 1], [0, 1], families.Binomial())
    result = check_zero_inflation(model)

    assert result.expected_zeros == pytest.approx(1)
    assert result.variance == pytest.approx(0)
    assert result.z_score == pytest.approx(0)
    assert result.p_value == pytest.approx(1)
    assert result.status == "ok"


@pytest.mark.parametrize("alpha", [0, 1, -0.1, np.nan])
def test_invalid_alpha_is_rejected(alpha: float) -> None:
    model = _GeneralizedModel([0, 1], [0.5, 1.0], families.Poisson())
    with pytest.raises(ValueError, match="alpha"):
        check_overdispersion(model, alpha=alpha)


def test_invalid_alternative_is_rejected() -> None:
    model = _GeneralizedModel([0, 1], [0.5, 1.0], families.Poisson())
    with pytest.raises(ValueError, match="alternative"):
        check_zero_inflation(model, alternative="invalid")


def test_linear_model_is_rejected() -> None:
    model = object.__new__(LmerResult)
    with pytest.raises(TypeError, match="generalized linear mixed model"):
        check_overdispersion(model)


def test_nonpositive_residual_df_is_rejected() -> None:
    model = _GeneralizedModel([0, 1], [0.5, 1.0], families.Poisson(), n_parameters=2)
    with pytest.raises(ValueError, match="degrees of freedom"):
        check_overdispersion(model)


def test_quasi_family_zero_check_is_rejected() -> None:
    model = _GeneralizedModel([0, 1], [0.5, 1.0], families.QuasiFamily(families.Poisson(), phi=2.0))
    with pytest.raises(TypeError, match="quasi-family"):
        check_zero_inflation(model)


def test_noninteger_binomial_trials_are_rejected() -> None:
    model = _GeneralizedModel([0, 1], [0.2, 0.8], families.Binomial(), weights=[1.5, 2.0])
    with pytest.raises(ValueError, match="integer trial-count"):
        check_zero_inflation(model)


def test_unsupported_zero_family_is_rejected() -> None:
    model = _GeneralizedModel([1, 2], [1, 2], families.Gamma())
    with pytest.raises(TypeError, match="support Poisson"):
        check_zero_inflation(model)


def test_fitted_poisson_model_diagnostics_are_finite() -> None:
    rng = np.random.default_rng(42)
    n_groups = 10
    n_per_group = 15
    group = np.repeat(np.arange(n_groups), n_per_group)
    x = rng.normal(size=group.size)
    random_effect = rng.normal(scale=0.3, size=n_groups)
    y = rng.poisson(np.exp(0.2 + 0.25 * x + random_effect[group]))
    data = pd.DataFrame({"y": y, "x": x, "group": group.astype(str)})
    model = glmer("y ~ x + (1 | group)", data, family=families.Poisson())

    dispersion = check_overdispersion(model)
    zeros = check_zero_inflation(model)

    assert dispersion.residual_df == len(data) - model.npar()
    assert np.isfinite(dispersion.dispersion_ratio)
    assert np.isfinite(zeros.expected_zeros)
    if zeros.expected_zeros > 0:
        assert_allclose(zeros.observed_to_expected, zeros.observed_zeros / zeros.expected_zeros)
    else:
        assert np.isinf(zeros.observed_to_expected)
