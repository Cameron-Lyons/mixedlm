"""Fast analytic diagnostics for generalized linear mixed models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from mixedlm.families import Binomial, NegativeBinomial, Poisson, QuasiFamily

if TYPE_CHECKING:
    from mixedlm.families import Family
    from mixedlm.models.glmer import GlmerResult

Alternative: TypeAlias = Literal["two-sided", "greater", "less"]
DiagnosticValue: TypeAlias = float | int | str

_VALID_ALTERNATIVES = {"two-sided", "greater", "less"}
_PROBABILITY_TOLERANCE = 1e-12


@dataclass(frozen=True)
class DispersionResult:
    """Result of a Pearson chi-squared dispersion check."""

    pearson_chi_square: float
    residual_df: int
    dispersion_ratio: float
    p_value: float
    ci_low: float
    ci_high: float
    alpha: float
    alternative: Alternative
    status: str

    @property
    def is_overdispersed(self) -> bool:
        """Whether statistically significant overdispersion was detected."""
        return self.status == "overdispersed"

    @property
    def is_underdispersed(self) -> bool:
        """Whether statistically significant underdispersion was detected."""
        return self.status == "underdispersed"

    def as_dict(self) -> dict[str, DiagnosticValue]:
        """Return the diagnostic as a plain dictionary."""
        return {
            "pearson_chi_square": self.pearson_chi_square,
            "residual_df": self.residual_df,
            "dispersion_ratio": self.dispersion_ratio,
            "p_value": self.p_value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "alpha": self.alpha,
            "alternative": self.alternative,
            "status": self.status,
        }

    def __str__(self) -> str:
        conclusion = {
            "overdispersed": "Overdispersion detected.",
            "underdispersed": "Underdispersion detected.",
            "ok": "No significant dispersion problem detected.",
        }[self.status]
        return "\n".join(
            [
                "Pearson dispersion check",
                f"  Dispersion ratio: {self.dispersion_ratio:.3f}",
                f"  Pearson chi-squared: {self.pearson_chi_square:.3f}",
                f"  Residual df: {self.residual_df}",
                f"  p-value ({self.alternative}): {self.p_value:.4g}",
                f"  {conclusion}",
            ]
        )


@dataclass(frozen=True)
class ZeroInflationResult:
    """Result of an analytic observed-versus-expected zero check."""

    observed_zeros: int
    expected_zeros: float
    observed_to_expected: float
    variance: float
    z_score: float
    p_value: float
    alpha: float
    alternative: Alternative
    status: str

    @property
    def is_zero_inflated(self) -> bool:
        """Whether a statistically significant excess of zeros was detected."""
        return self.status == "excess"

    def as_dict(self) -> dict[str, DiagnosticValue]:
        """Return the diagnostic as a plain dictionary."""
        return {
            "observed_zeros": self.observed_zeros,
            "expected_zeros": self.expected_zeros,
            "observed_to_expected": self.observed_to_expected,
            "variance": self.variance,
            "z_score": self.z_score,
            "p_value": self.p_value,
            "alpha": self.alpha,
            "alternative": self.alternative,
            "status": self.status,
        }

    def __str__(self) -> str:
        conclusion = {
            "excess": "Excess zeros detected.",
            "deficit": "Fewer zeros than expected detected.",
            "ok": "No significant zero-count mismatch detected.",
        }[self.status]
        return "\n".join(
            [
                "Zero-inflation check",
                f"  Observed zeros: {self.observed_zeros}",
                f"  Expected zeros: {self.expected_zeros:.3f}",
                f"  Observed / expected: {self.observed_to_expected:.3f}",
                f"  p-value ({self.alternative}): {self.p_value:.4g}",
                f"  {conclusion}",
            ]
        )


def _validate_options(alpha: float, alternative: str) -> Alternative:
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be a finite number strictly between 0 and 1")
    if alternative not in _VALID_ALTERNATIVES:
        choices = ", ".join(sorted(_VALID_ALTERNATIVES))
        raise ValueError(f"alternative must be one of: {choices}")
    return cast(Alternative, alternative)


def _validate_model(
    model: GlmerResult,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], Family]:
    is_glmm = getattr(model, "isGLMM", None)
    if not callable(is_glmm) or not is_glmm():
        raise TypeError("diagnostic requires a fitted generalized linear mixed model")

    y = np.asarray(model.matrices.y, dtype=np.float64)
    mu = np.asarray(model.fitted(type="response", na_expand=False), dtype=np.float64)
    weights = np.asarray(model.matrices.weights, dtype=np.float64)
    if y.ndim != 1 or mu.shape != y.shape or weights.shape != y.shape:
        raise ValueError(
            "response, fitted values, and weights must be aligned one-dimensional arrays"
        )
    if y.size == 0:
        raise ValueError("diagnostic requires at least one observation")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(mu)):
        raise ValueError("response and fitted values must be finite")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("model weights must be finite and strictly positive")
    return y, mu, weights, model.family


def _tail_probability(cdf: float, sf: float, alternative: Alternative) -> float:
    if alternative == "greater":
        return sf
    if alternative == "less":
        return cdf
    return min(1.0, 2 * min(cdf, sf))


def _dispersion_status(ratio: float, p_value: float, alpha: float, alternative: Alternative) -> str:
    if p_value >= alpha:
        return "ok"
    if alternative == "greater":
        return "overdispersed"
    if alternative == "less":
        return "underdispersed"
    return "overdispersed" if ratio > 1 else "underdispersed"


def check_overdispersion(
    model: GlmerResult,
    *,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> DispersionResult:
    """Check conditional dispersion with a weighted Pearson chi-squared test.

    The test is an analytic approximation. It is most useful for Poisson and
    grouped-binomial models; simulation-based residual checks are preferable
    for small means, Bernoulli responses, and complex variance structures.

    Parameters
    ----------
    model : GlmerResult
        Fitted generalized linear mixed model.
    alpha : float, default 0.05
        Significance level used for the conclusion and confidence interval.
    alternative : {"two-sided", "greater", "less"}, default "two-sided"
        Whether to test for any mismatch, overdispersion, or underdispersion.

    Returns
    -------
    DispersionResult
        Pearson statistic, residual degrees of freedom, dispersion ratio,
        confidence interval, p-value, and conclusion.
    """
    alternative = _validate_options(alpha, alternative)
    y, mu, weights, family = _validate_model(model)

    variance = np.asarray(family.variance(mu), dtype=np.float64)
    if variance.shape != y.shape:
        raise ValueError("family variance must have the same shape as the fitted values")
    if not np.all(np.isfinite(variance)) or np.any(variance <= 0):
        raise ValueError("family variance must be finite and strictly positive")

    residual_df = y.size - int(model.npar())
    if residual_df <= 0:
        raise ValueError("residual degrees of freedom must be positive")

    pearson = float(np.sum(weights * np.square(y - mu) / variance))
    ratio = pearson / residual_df
    cdf = float(stats.chi2.cdf(pearson, residual_df))
    sf = float(stats.chi2.sf(pearson, residual_df))
    p_value = _tail_probability(cdf, sf, alternative)

    lower_quantile = float(stats.chi2.ppf(1 - alpha / 2, residual_df))
    upper_quantile = float(stats.chi2.ppf(alpha / 2, residual_df))
    ci_low = pearson / lower_quantile
    ci_high = pearson / upper_quantile
    status = _dispersion_status(ratio, p_value, alpha, alternative)

    return DispersionResult(
        pearson_chi_square=pearson,
        residual_df=residual_df,
        dispersion_ratio=ratio,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        alpha=alpha,
        alternative=alternative,
        status=status,
    )


def _zero_probabilities(
    family: Family,
    y: NDArray[np.float64],
    mu: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    if isinstance(family, QuasiFamily):
        raise TypeError(
            "zero-inflation checks require a full probability distribution, not a quasi-family"
        )

    if isinstance(family, Poisson):
        if np.any(mu < 0) or np.any(y < 0) or not np.allclose(y, np.round(y), rtol=0, atol=1e-8):
            raise ValueError("Poisson responses must be non-negative integer counts")
        return np.exp(-mu)

    if isinstance(family, NegativeBinomial):
        if np.any(mu < 0) or np.any(y < 0) or not np.allclose(y, np.round(y), rtol=0, atol=1e-8):
            raise ValueError("negative-binomial responses must be non-negative integer counts")
        theta = float(family.theta)
        if not np.isfinite(theta) or theta <= 0:
            raise ValueError("negative-binomial theta must be finite and strictly positive")
        return np.exp(-theta * np.log1p(mu / theta))

    if isinstance(family, Binomial):
        trials = np.round(weights)
        if not np.allclose(weights, trials, rtol=0, atol=1e-8):
            raise ValueError("binomial zero checks require integer trial-count weights")
        if np.any(mu < 0) or np.any(mu > 1) or np.any(y < 0) or np.any(y > 1):
            raise ValueError("binomial responses and fitted probabilities must be between 0 and 1")
        if not np.allclose(y * trials, np.round(y * trials), rtol=0, atol=1e-8):
            raise ValueError("weighted binomial responses must represent integer success counts")
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.exp(trials * np.log1p(-mu))

    raise TypeError(
        "zero-inflation checks support Poisson, negative-binomial, and binomial families"
    )


def _zero_tail_probabilities(
    observed: int, expected: float, variance: float
) -> tuple[float, float, float]:
    if variance <= _PROBABILITY_TOLERANCE:
        difference = observed - expected
        z_score = (
            0.0
            if abs(difference) <= _PROBABILITY_TOLERANCE
            else float(np.sign(difference) * np.inf)
        )
        p_greater = 0.0 if difference > _PROBABILITY_TOLERANCE else 1.0
        p_less = 0.0 if difference < -_PROBABILITY_TOLERANCE else 1.0
        return z_score, p_greater, p_less

    sd = np.sqrt(variance)
    z_score = (observed - expected) / sd
    p_greater = float(stats.norm.sf((observed - 0.5 - expected) / sd))
    p_less = float(stats.norm.cdf((observed + 0.5 - expected) / sd))
    return float(z_score), p_greater, p_less


def _zero_status(difference: float, p_value: float, alpha: float, alternative: Alternative) -> str:
    if p_value >= alpha:
        return "ok"
    if alternative == "greater":
        return "excess"
    if alternative == "less":
        return "deficit"
    return "excess" if difference > 0 else "deficit"


def check_zero_inflation(
    model: GlmerResult,
    *,
    alpha: float = 0.05,
    alternative: Alternative = "greater",
) -> ZeroInflationResult:
    """Compare observed zeros with the fitted response distribution.

    The zero count is approximated by a continuity-corrected normal
    distribution with the exact Poisson-binomial mean and variance. The check
    is conditional on the fitted random effects and ignores parameter-estimation
    uncertainty.

    Parameters
    ----------
    model : GlmerResult
        Fitted Poisson, negative-binomial, or binomial mixed model.
    alpha : float, default 0.05
        Significance level used for the conclusion.
    alternative : {"two-sided", "greater", "less"}, default "greater"
        Whether to test for any mismatch, excess zeros, or a deficit of zeros.

    Returns
    -------
    ZeroInflationResult
        Observed and expected counts, their ratio, approximate p-value, and
        conclusion.
    """
    alternative = _validate_options(alpha, alternative)
    y, mu, weights, family = _validate_model(model)
    zero_probabilities = _zero_probabilities(family, y, mu, weights)
    if (
        not np.all(np.isfinite(zero_probabilities))
        or np.any(zero_probabilities < 0)
        or np.any(zero_probabilities > 1)
    ):
        raise ValueError("fitted zero probabilities must be finite and between 0 and 1")

    observed = int(np.count_nonzero(y == 0))
    expected = float(np.sum(zero_probabilities))
    variance = float(np.sum(zero_probabilities * (1 - zero_probabilities)))
    ratio = observed / expected if expected > 0 else (1.0 if observed == 0 else float("inf"))

    z_score, p_greater, p_less = _zero_tail_probabilities(observed, expected, variance)
    p_value = _tail_probability(p_less, p_greater, alternative)
    status = _zero_status(observed - expected, p_value, alpha, alternative)

    return ZeroInflationResult(
        observed_zeros=observed,
        expected_zeros=expected,
        observed_to_expected=ratio,
        variance=variance,
        z_score=z_score,
        p_value=p_value,
        alpha=alpha,
        alternative=alternative,
        status=status,
    )


# Compact spelling retained for users coming from the R diagnostics ecosystem.
check_zeroinflation = check_zero_inflation


__all__ = [
    "DispersionResult",
    "ZeroInflationResult",
    "check_overdispersion",
    "check_zero_inflation",
    "check_zeroinflation",
]
