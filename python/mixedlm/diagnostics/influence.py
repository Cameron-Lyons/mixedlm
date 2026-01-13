from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import linalg

if TYPE_CHECKING:
    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult


@dataclass
class InfluenceResult:
    hat_values: NDArray[np.floating]
    residuals: NDArray[np.floating]
    sigma: float
    X: NDArray[np.floating]
    beta: NDArray[np.floating]
    vcov: NDArray[np.floating]
    model_type: str

    @property
    def dfbeta(self) -> NDArray[np.floating]:
        XtX_inv = self.vcov / (self.sigma**2)
        n = len(self.residuals)
        dfb = np.zeros((n, len(self.beta)))

        for i in range(n):
            h_i = self.hat_values[i]
            r_i = self.residuals[i]
            x_i = self.X[i, :]
            if h_i < 1.0 - 1e-10:
                dfb[i, :] = (XtX_inv @ x_i) * r_i / (1 - h_i)

        return dfb

    @property
    def dfbetas(self) -> NDArray[np.floating]:
        se = np.sqrt(np.diag(self.vcov))
        dfb = self.dfbeta
        return dfb / se

    @property
    def cooks_distance(self) -> NDArray[np.floating]:
        p = len(self.beta)
        h = self.hat_values
        r = self.residuals

        standardized_resid = r / (self.sigma * np.sqrt(1 - h + 1e-10))
        cooks = (standardized_resid**2 / p) * (h / (1 - h + 1e-10))

        return cooks

    @property
    def dffits(self) -> NDArray[np.floating]:
        h = self.hat_values
        r = self.residuals
        studentized = r / (self.sigma * np.sqrt(1 - h + 1e-10))
        return studentized * np.sqrt(h / (1 - h + 1e-10))


def influence(
    model: LmerResult | GlmerResult,
) -> InfluenceResult:
    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult

    if isinstance(model, LmerResult):
        return _influence_lmer(model)
    elif isinstance(model, GlmerResult):
        return _influence_glmer(model)
    else:
        raise TypeError(f"Unsupported model type: {type(model)}")


def _influence_lmer(model: LmerResult) -> InfluenceResult:
    X = model.matrices.X
    y = model.matrices.y
    beta = model.beta
    sigma = model.sigma

    fitted = X @ beta
    residuals = y - fitted

    XtX = X.T @ X
    try:
        XtX_inv = linalg.inv(XtX)
    except linalg.LinAlgError:
        XtX_inv = linalg.pinv(XtX)

    H = X @ XtX_inv @ X.T
    hat_values = np.diag(H)

    vcov = sigma**2 * XtX_inv

    return InfluenceResult(
        hat_values=hat_values,
        residuals=residuals,
        sigma=sigma,
        X=X,
        beta=beta,
        vcov=vcov,
        model_type="lmer",
    )


def _influence_glmer(model: GlmerResult) -> InfluenceResult:
    X = model.matrices.X
    y = model.matrices.y
    beta = model.beta

    eta = X @ beta
    mu = model.family.link.inverse(eta)
    residuals = y - mu

    weights = model.family.variance(mu)
    W = np.diag(weights)

    XtWX = X.T @ W @ X
    try:
        XtWX_inv = linalg.inv(XtWX)
    except linalg.LinAlgError:
        XtWX_inv = linalg.pinv(XtWX)

    W_sqrt = np.diag(np.sqrt(weights))
    H = W_sqrt @ X @ XtWX_inv @ X.T @ W_sqrt
    hat_values = np.diag(H)

    scale = 1.0
    if hasattr(model.family, "scale") and model.family.scale is not None:
        scale = model.family.scale

    vcov = scale * XtWX_inv

    return InfluenceResult(
        hat_values=hat_values,
        residuals=residuals,
        sigma=np.sqrt(scale),
        X=X,
        beta=beta,
        vcov=vcov,
        model_type="glmer",
    )


def dfbeta(inf: InfluenceResult) -> NDArray[np.floating]:
    return inf.dfbeta


def dfbetas(inf: InfluenceResult) -> NDArray[np.floating]:
    return inf.dfbetas


def cooks_distance(inf: InfluenceResult) -> NDArray[np.floating]:
    return inf.cooks_distance


def dffits(inf: InfluenceResult) -> NDArray[np.floating]:
    return inf.dffits


def leverage(inf: InfluenceResult) -> NDArray[np.floating]:
    return inf.hat_values


def influence_plot(
    inf: InfluenceResult,
    which: str = "cooks",
    ax=None,
):
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


def influence_summary(inf: InfluenceResult) -> pd.DataFrame:
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
    inf: InfluenceResult,
    threshold: str = "cooks",
) -> NDArray[np.intp]:
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
