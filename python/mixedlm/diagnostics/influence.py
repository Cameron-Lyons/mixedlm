from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pandas as pd

    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult


def _cooks_distance(
    residuals: NDArray[np.floating],
    hat_values: NDArray[np.floating],
    n_params: int,
    sigma: float,
) -> NDArray[np.floating]:
    """Cook's distance ``r**2 / (p * sigma**2) * h / (1 - h)**2``; NaN when ``p == 0``."""
    if n_params == 0:
        # The normalization divides by the number of coefficients.
        return np.full(len(residuals), np.nan)
    h = np.clip(hat_values, 0, 1 - 1e-10)
    return (residuals**2 / (n_params * sigma**2)) * (h / (1 - h) ** 2)


@dataclass
class InfluenceResult:
    """Influence measures evaluated on the fitted model's working scale.

    Linear-model residuals and design rows include square-root prior weights.
    Generalized-model values use the final IRLS working projection.
    Coefficient deletion diagnostics hold variance components fixed; generalized
    models also hold the final working weights fixed. They account for the
    random effects when computing changes to fixed-effect coefficients.
    """

    hat_values: NDArray[np.floating]
    residuals: NDArray[np.floating]
    sigma: float
    X: NDArray[np.floating]
    beta: NDArray[np.floating]
    vcov: NDArray[np.floating]
    model_type: str
    # Deferred because only coefficient-deletion measures need the n x p product.
    _beta_sensitivity_factory: Callable[[], NDArray[np.floating]] | None = field(
        default=None, repr=False
    )

    @cached_property
    def _beta_sensitivity(self) -> NDArray[np.floating] | None:
        factory = self._beta_sensitivity_factory
        return None if factory is None else factory()

    @property
    def leverage(self) -> NDArray[np.floating]:
        """Compatibility alias for the leverage values."""
        return self.hat_values

    @property
    def dfbeta(self) -> NDArray[np.floating]:
        h = self.hat_values
        r = self.residuals
        denom = 1 - h
        denom = np.where(denom > 1e-10, denom, np.inf)
        scale = r / denom

        if self._beta_sensitivity is not None:
            return self._beta_sensitivity * scale[:, None]

        XtX_inv = self.vcov / (self.sigma**2)
        return (self.X @ XtX_inv) * scale[:, None]

    @property
    def dfbetas(self) -> NDArray[np.floating]:
        se = np.sqrt(np.diag(self.vcov))
        dfb = self.dfbeta
        return dfb / se

    @property
    def cooks_distance(self) -> NDArray[np.floating]:
        return _cooks_distance(self.residuals, self.hat_values, len(self.beta), self.sigma)

    @property
    def dffits(self) -> NDArray[np.floating]:
        h = np.clip(self.hat_values, 0, 1 - 1e-10)
        r = self.residuals
        return (r / self.sigma) * np.sqrt(h) / (1 - h)


def influence(
    model: LmerResult | GlmerResult,
) -> InfluenceResult:
    """Compute influence diagnostics from a fitted mixed model."""
    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult

    if isinstance(model, LmerResult):
        return _influence_lmer(model)
    elif isinstance(model, GlmerResult):
        return _influence_glmer(model)
    else:
        raise TypeError(f"Unsupported model type: {type(model)}")


def _influence_lmer(model: LmerResult) -> InfluenceResult:
    projection = model._weighted_projection
    vcov = model.vcov()
    residuals = projection.sqrt_weights * model.residuals(type="response", na_expand=False)

    def beta_sensitivity() -> NDArray[np.floating]:
        adjusted_X = projection.weighted_X
        if projection.lambda_matrix is not None:
            weighted_random = projection.weighted_Z @ projection.lambda_matrix
            adjusted_X = adjusted_X - weighted_random @ projection.random_fixed_map
        return adjusted_X @ (vcov / model.sigma**2)

    return InfluenceResult(
        hat_values=model.hatvalues(),
        residuals=residuals,
        sigma=model.sigma,
        X=projection.weighted_X,
        beta=model.beta,
        vcov=vcov,
        model_type="lmer",
        _beta_sensitivity_factory=beta_sensitivity,
    )


def _influence_glmer(model: GlmerResult) -> InfluenceResult:
    projection = model._working_projection
    vcov = model.vcov()
    mu = model.family.clamp_mu(model.fitted(type="response", na_expand=False))
    # Convert Pearson residuals to the final weighted working residual. This
    # retains the derivative's sign for decreasing links and the actual working
    # weights when PIRLS clips them at numerical bounds.
    working_scale = (
        np.sqrt(projection.weights)
        * model.family.link.deriv(mu)
        * np.sqrt(model.family.variance(mu) / model.matrices.weights)
    )

    def beta_sensitivity() -> NDArray[np.floating]:
        adjusted_X = projection.weighted_X
        if model.matrices.n_random:
            weighted_random = projection.weighted_Z @ projection.Lambda
            adjusted_X = adjusted_X - weighted_random @ projection.random_fixed_map
        return (adjusted_X @ vcov) * working_scale[:, None]

    return InfluenceResult(
        hat_values=model.hatvalues(),
        residuals=model.residuals(type="pearson", na_expand=False),
        sigma=1.0,
        X=projection.weighted_X,
        beta=model.beta,
        vcov=vcov,
        model_type="glmer",
        _beta_sensitivity_factory=beta_sensitivity,
    )


def _coerce_influence(
    value: InfluenceResult | LmerResult | GlmerResult,
) -> InfluenceResult:
    return value if isinstance(value, InfluenceResult) else influence(value)


def dfbeta(inf: InfluenceResult | LmerResult | GlmerResult) -> NDArray[np.floating]:
    return _coerce_influence(inf).dfbeta


def dfbetas(inf: InfluenceResult | LmerResult | GlmerResult) -> NDArray[np.floating]:
    return _coerce_influence(inf).dfbetas


def cooks_distance(inf: InfluenceResult | LmerResult | GlmerResult) -> NDArray[np.floating]:
    return _coerce_influence(inf).cooks_distance


def dffits(inf: InfluenceResult | LmerResult | GlmerResult) -> NDArray[np.floating]:
    return _coerce_influence(inf).dffits


def leverage(inf: InfluenceResult | LmerResult | GlmerResult) -> NDArray[np.floating]:
    return _coerce_influence(inf).hat_values


def influence_plot(
    inf: InfluenceResult | LmerResult | GlmerResult,
    which: str = "cooks",
    ax=None,
):
    inf = _coerce_influence(inf)
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError("matplotlib is required for influence_plot") from err

    if ax is None:
        _, ax = plt.subplots()

    if which == "cooks":
        values = inf.cooks_distance
        ylabel = "Cook's Distance"
    elif which == "dfbetas":
        values = np.max(np.abs(inf.dfbetas), axis=1)
        ylabel = "Max |DFBETAS|"
    elif which == "leverage":
        values = inf.hat_values
        ylabel = "Leverage (Hat Values)"
    elif which == "dffits":
        values = np.abs(inf.dffits)
        ylabel = "|DFFITS|"
    else:
        raise ValueError(f"Unknown plot type: {which}")

    n = len(values)
    ax.stem(range(n), values)
    ax.set_xlabel("Observation")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Influence Diagnostics ({which})")

    if which == "cooks":
        threshold = 4 / n
    elif which == "dfbetas":
        threshold = 2 / np.sqrt(n)
    elif which == "leverage":
        p = len(inf.beta)
        threshold = 2 * p / n
    elif which == "dffits":
        p = len(inf.beta)
        threshold = 2 * np.sqrt(p / n)
    else:
        threshold = None

    if threshold is not None:
        ax.axhline(y=threshold, color="r", linestyle="--", alpha=0.5, label="threshold")
        ax.legend()


def influence_summary(
    inf: InfluenceResult | LmerResult | GlmerResult,
) -> pd.DataFrame:
    import pandas as pd

    inf = _coerce_influence(inf)
    cooks = inf.cooks_distance
    max_dfbetas = np.max(np.abs(inf.dfbetas), axis=1)
    leverage = inf.hat_values
    dffits_vals = np.abs(inf.dffits)

    n = len(cooks)
    p = len(inf.beta)

    df = pd.DataFrame(
        {
            "observation": range(n),
            "cooks_distance": cooks,
            "max_abs_dfbetas": max_dfbetas,
            "leverage": leverage,
            "abs_dffits": dffits_vals,
        }
    )

    cooks_threshold = 4 / n
    dfbetas_threshold = 2 / np.sqrt(n)
    leverage_threshold = 2 * p / n
    dffits_threshold = 2 * np.sqrt(p / n)

    df["influential_cooks"] = df["cooks_distance"] > cooks_threshold
    df["influential_dfbetas"] = df["max_abs_dfbetas"] > dfbetas_threshold
    df["high_leverage"] = df["leverage"] > leverage_threshold
    df["influential_dffits"] = df["abs_dffits"] > dffits_threshold

    return df.sort_values("cooks_distance", ascending=False)


def influential_obs(
    inf: InfluenceResult | LmerResult | GlmerResult,
    threshold: str = "cooks",
) -> NDArray[np.intp]:
    inf = _coerce_influence(inf)
    n = len(inf.residuals)
    p = len(inf.beta)

    if threshold == "cooks":
        values = inf.cooks_distance
        cutoff = 4 / n
    elif threshold == "dfbetas":
        values = np.max(np.abs(inf.dfbetas), axis=1)
        cutoff = 2 / np.sqrt(n)
    elif threshold == "leverage":
        values = inf.hat_values
        cutoff = 2 * p / n
    elif threshold == "dffits":
        values = np.abs(inf.dffits)
        cutoff = 2 * np.sqrt(p / n)
    else:
        raise ValueError(f"Unknown threshold type: {threshold}")

    return np.where(values > cutoff)[0]
