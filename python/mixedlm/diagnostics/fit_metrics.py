from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mixedlm.models.shared_utils import dense_quadratic_form_diagonal


@dataclass(frozen=True)
class R2NakagawaResult:
    """Marginal and conditional R-squared with their variance decomposition."""

    marginal: float
    conditional: float
    variance_fixed: float
    variance_random: float
    variance_residual: float
    random_by_group: dict[str, float]
    approximation: str

    def as_dict(self) -> dict[str, float]:
        """Return the scalar metrics as a dictionary."""
        return {
            "marginal": self.marginal,
            "conditional": self.conditional,
            "variance_fixed": self.variance_fixed,
            "variance_random": self.variance_random,
            "variance_residual": self.variance_residual,
        }

    def __str__(self) -> str:
        return f"Nakagawa R²: marginal={self.marginal:.4f}, conditional={self.conditional:.4f}"


@dataclass(frozen=True)
class ICCResult:
    """Adjusted and unadjusted intraclass correlation coefficients."""

    adjusted: float
    unadjusted: float
    by_group: dict[str, float]
    by_group_unadjusted: dict[str, float]
    variance_fixed: float
    variance_random: float
    variance_residual: float
    approximation: str

    def as_dict(self) -> dict[str, float]:
        """Return the scalar metrics as a dictionary."""
        return {
            "adjusted": self.adjusted,
            "unadjusted": self.unadjusted,
            "variance_fixed": self.variance_fixed,
            "variance_random": self.variance_random,
            "variance_residual": self.variance_residual,
        }

    def __str__(self) -> str:
        return f"ICC: adjusted={self.adjusted:.4f}, unadjusted={self.unadjusted:.4f}"


@dataclass(frozen=True)
class _VarianceComponents:
    fixed: float
    random: float
    residual: float
    random_by_group: dict[str, float]
    approximation: str


def r2_nakagawa(model: Any, approximation: str = "lognormal") -> R2NakagawaResult:
    """Compute Nakagawa marginal and conditional R-squared.

    Marginal R-squared is the share of total variance explained by fixed
    effects. Conditional R-squared includes both fixed and random effects.
    Random-slope variance is evaluated for every observation before being
    averaged, so covariance and predictor values are retained.

    Parameters
    ----------
    model : LmerResult, GlmerResult, or NlmerResult
        Fitted mixed model.
    approximation : {"lognormal", "delta", "theoretical"}, default "lognormal"
        Link-scale residual variance approximation for generalized models.
        Lognormal applies to log links, while binomial models use their
        link-specific theoretical variance. Other links fall back to the
        delta method. Linear and nonlinear Gaussian models average the
        observation-specific residual variance ``sigma**2 / weights``.

    Returns
    -------
    R2NakagawaResult
        R-squared values and the full variance decomposition.
    """
    components = _variance_components(model, approximation)
    total = components.fixed + components.random + components.residual
    marginal = _safe_ratio(components.fixed, total)
    conditional = _safe_ratio(components.fixed + components.random, total)
    return R2NakagawaResult(
        marginal=marginal,
        conditional=conditional,
        variance_fixed=components.fixed,
        variance_random=components.random,
        variance_residual=components.residual,
        random_by_group=components.random_by_group,
        approximation=components.approximation,
    )


def icc(model: Any, approximation: str = "lognormal") -> ICCResult:
    """Compute adjusted and unadjusted intraclass correlations.

    The adjusted ICC excludes fixed-effect variance from its denominator.
    The unadjusted ICC uses the complete fixed, random, and residual variance.
    Group-specific values partition the corresponding overall ICC.
    """
    components = _variance_components(model, approximation)
    adjusted_denominator = components.random + components.residual
    total = components.fixed + adjusted_denominator
    adjusted = _safe_ratio(components.random, adjusted_denominator)
    unadjusted = _safe_ratio(components.random, total)
    by_group = {
        group: _safe_ratio(variance, adjusted_denominator)
        for group, variance in components.random_by_group.items()
    }
    by_group_unadjusted = {
        group: _safe_ratio(variance, total)
        for group, variance in components.random_by_group.items()
    }
    return ICCResult(
        adjusted=adjusted,
        unadjusted=unadjusted,
        by_group=by_group,
        by_group_unadjusted=by_group_unadjusted,
        variance_fixed=components.fixed,
        variance_random=components.random,
        variance_residual=components.residual,
        approximation=components.approximation,
    )


def _variance_components(model: Any, approximation: str) -> _VarianceComponents:
    method = approximation.strip().lower()
    if method not in ("lognormal", "delta", "theoretical"):
        raise ValueError(
            f"Unknown approximation '{approximation}'. Use 'lognormal', 'delta', or 'theoretical'."
        )

    if _model_flag(model, "isLMM"):
        return _linear_variance_components(model)
    if _model_flag(model, "isGLMM"):
        return _generalized_variance_components(model, method)
    if _model_flag(model, "isNLMM"):
        return _nonlinear_variance_components(model)
    raise TypeError("Expected a fitted linear, generalized, or nonlinear mixed model")


def _linear_variance_components(model: Any) -> _VarianceComponents:
    fixed, random, by_group = _linear_predictor_variances(model, scale=model.sigma**2)
    return _VarianceComponents(
        fixed=fixed,
        random=random,
        residual=_gaussian_residual_variance(model.sigma, model.matrices.weights),
        random_by_group=by_group,
        approximation="gaussian",
    )


def _generalized_variance_components(
    model: Any,
    approximation: str,
) -> _VarianceComponents:
    fixed, random, by_group = _linear_predictor_variances(model, scale=1.0)
    eta_fixed = _fixed_linear_predictor(model)
    random_predictor = np.asarray(model.matrices.Z @ model.u, dtype=np.float64).reshape(-1)
    mu = np.asarray(model.family.link.inverse(eta_fixed + random_predictor), dtype=np.float64)
    residual, used_approximation = _distribution_specific_variance(
        model.family,
        mu,
        np.asarray(model.matrices.weights, dtype=np.float64),
        approximation,
    )
    return _VarianceComponents(
        fixed=fixed,
        random=random,
        residual=residual,
        random_by_group=by_group,
        approximation=used_approximation,
    )


def _nonlinear_variance_components(model: Any) -> _VarianceComponents:
    from mixedlm.estimation.nlmm import _build_psi_matrix

    fixed_predictions = np.asarray(model.model.predict(model.phi, model.x), dtype=np.float64)
    fixed_predictions = fixed_predictions + model.offset(copy=False)
    fixed = _sample_variance(fixed_predictions)
    n_random = len(model.random_params)
    if n_random:
        gradient = np.asarray(model.model.gradient(model.phi, model.x), dtype=np.float64)
        random_gradient = gradient[:, model.random_params]
        covariance = _build_psi_matrix(model.theta, n_random) * model.sigma**2
        row_variance = np.sum((random_gradient @ covariance) * random_gradient, axis=1)
        random = _nonnegative_mean(row_variance)
    else:
        random = 0.0
    by_group = {model.group_var: random} if random > 0.0 else {}
    return _VarianceComponents(
        fixed=fixed,
        random=random,
        residual=_gaussian_residual_variance(model.sigma, model.weights(copy=False)),
        random_by_group=by_group,
        approximation="gaussian-delta",
    )


def _linear_predictor_variances(
    model: Any,
    scale: float,
) -> tuple[float, float, dict[str, float]]:
    fixed = _sample_variance(_fixed_linear_predictor(model))
    by_group: dict[str, float] = {}
    column_start = 0

    for structure, covariance in model._iter_random_cov_blocks(scale=scale):
        n_terms = structure.n_terms
        width = structure.n_levels * n_terms
        block = model.matrices.Z[:, column_start : column_start + width]
        term_design = np.column_stack(
            [np.asarray(block[:, term::n_terms].sum(axis=1)).reshape(-1) for term in range(n_terms)]
        )
        covariance_array = np.asarray(covariance, dtype=np.float64)
        row_variance = dense_quadratic_form_diagonal(term_design, covariance_array)
        component = _nonnegative_mean(row_variance)
        by_group[structure.grouping_factor] = (
            by_group.get(structure.grouping_factor, 0.0) + component
        )
        column_start += width

    if column_start != model.matrices.Z.shape[1]:
        raise ValueError(
            f"Random-effect structures describe {column_start} columns, "
            f"but Z has {model.matrices.Z.shape[1]}"
        )
    return fixed, float(sum(by_group.values())), by_group


def _fixed_linear_predictor(model: Any) -> NDArray[np.float64]:
    predictor = np.asarray(model.matrices.X @ model.beta, dtype=np.float64).reshape(-1)
    offset = np.asarray(model.matrices.offset, dtype=np.float64).reshape(-1)
    if len(offset) != len(predictor):
        raise ValueError(f"offset has length {len(offset)}, expected {len(predictor)}")
    return predictor + offset


def _distribution_specific_variance(
    family: Any,
    mu: NDArray[np.float64],
    weights: NDArray[np.float64],
    approximation: str,
) -> tuple[float, str]:
    from mixedlm.families.base import (
        CloglogLink,
        LogitLink,
        LogLink,
        ProbitLink,
    )
    from mixedlm.families.binomial import Binomial

    if len(weights) != len(mu):
        raise ValueError(f"weights has length {len(weights)}, expected {len(mu)}")
    if np.any(~np.isfinite(mu)):
        raise ValueError("Fitted means must be finite to compute generalized-model R-squared")
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("Model weights must be finite and positive")

    base_family = getattr(family, "base_family", family)
    if isinstance(base_family, Binomial) and approximation != "delta":
        dispersion = float(getattr(family, "phi", 1.0))
        if isinstance(family.link, LogitLink):
            return dispersion * np.pi**2 / 3.0, "theoretical"
        if isinstance(family.link, ProbitLink):
            return dispersion, "theoretical"
        if isinstance(family.link, CloglogLink):
            return dispersion * np.pi**2 / 6.0, "theoretical"

    if approximation == "theoretical":
        raise ValueError("The theoretical approximation is only available for binomial links")

    if (
        isinstance(family.link, LogLink)
        and type(family.link).deriv is LogLink.deriv
        and type(family.link).link is LogLink.link
        and type(family.link).inverse is LogLink.inverse
    ):
        log_relative = _log_relative_variance(family, mu, weights)
        if log_relative is not None:
            if approximation == "lognormal":
                return _nonnegative_mean(np.logaddexp(0.0, log_relative)), "lognormal"
            with np.errstate(over="ignore"):
                values = np.exp(log_relative)
            if np.any(~np.isfinite(values)):
                raise ValueError("Delta-method residual variance is not finite")
            return _nonnegative_mean(values), "delta"

    conditional_variance = np.asarray(family.variance(mu), dtype=np.float64) / weights
    if np.any(~np.isfinite(conditional_variance)) or np.any(conditional_variance < 0.0):
        raise ValueError("Family variance must be finite and nonnegative")

    if approximation == "lognormal" and isinstance(family.link, LogLink):
        with np.errstate(divide="ignore", invalid="ignore"):
            log_relative = np.log(conditional_variance) - 2 * np.log(mu)
        values = np.logaddexp(0.0, log_relative)
        return _nonnegative_mean(values), "lognormal"

    derivative = np.asarray(family.link.deriv(mu), dtype=np.float64)
    values = derivative**2 * conditional_variance
    if np.any(~np.isfinite(values)):
        raise ValueError("Delta-method residual variance is not finite")
    return _nonnegative_mean(values), "delta"


def _log_relative_variance(
    family: Any, mu: NDArray[np.float64], weights: NDArray[np.float64]
) -> NDArray[np.float64] | None:
    """Evaluate log(V(mu) / (weight * mu²)) without forming large powers."""
    from mixedlm.families import Gamma, Gaussian, InverseGaussian, NegativeBinomial, Poisson
    from mixedlm.families.custom import QuasiFamily

    base_family = getattr(family, "base_family", family)
    if family is not base_family and (
        not isinstance(family, QuasiFamily) or type(family).variance is not QuasiFamily.variance
    ):
        return None
    variance_method = type(base_family).variance
    if variance_method not in (
        Gaussian.variance,
        Poisson.variance,
        Gamma.variance,
        InverseGaussian.variance,
        NegativeBinomial.variance,
    ):
        # Custom variance overrides retain their own mean-variance contract.
        return None
    if np.any(mu <= 0):
        raise ValueError("Log-link fitted means must be positive")
    log_mu = np.log(mu)
    if variance_method is Gaussian.variance:
        values = -2 * log_mu
    elif variance_method is Poisson.variance:
        values = -log_mu
    elif variance_method is Gamma.variance:
        values = np.zeros_like(mu)
    elif variance_method is InverseGaussian.variance:
        values = log_mu
    else:
        values = np.logaddexp(-log_mu, -np.log(base_family.theta))
    values -= np.log(weights)
    if family is not base_family:
        values += np.log(family.phi)
    return values


def _sample_variance(values: NDArray[np.float64]) -> float:
    if np.any(~np.isfinite(values)):
        raise ValueError("Fixed predictions must be finite")
    if len(values) <= 1:
        return 0.0
    # Center first so a large common baseline does not erase small differences.
    return max(float(np.var(values - values[0], ddof=1)), 0.0)


def _nonnegative_mean(values: NDArray[np.float64]) -> float:
    if np.any(~np.isfinite(values)):
        raise ValueError("Variance contributions must be finite")
    return max(float(np.mean(values)), 0.0) if len(values) else 0.0


def _gaussian_residual_variance(sigma: float, weights: NDArray[np.float64]) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Model weights must be finite and positive")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("Residual scale must be finite and nonnegative")
    result = float(sigma**2 * np.mean(1.0 / weights))
    if not np.isfinite(result):
        raise ValueError("Residual variance exceeds the floating-point range")
    return result


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan")
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def _model_flag(model: Any, name: str) -> bool:
    flag = getattr(model, name, None)
    return bool(flag()) if callable(flag) else False
