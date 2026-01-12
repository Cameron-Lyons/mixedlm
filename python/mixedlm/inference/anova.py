from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from mixedlm.models.lmer import LmerResult
    from mixedlm.models.glmer import GlmerResult


@dataclass
class AnovaResult:
    models: list[str]
    n_obs: list[int]
    df: list[int]
    aic: list[float]
    bic: list[float]
    loglik: list[float]
    deviance: list[float]
    chi_sq: list[float | None]
    chi_df: list[int | None]
    p_value: list[float | None]

    def __str__(self) -> str:
        lines = []
        lines.append("Data: model comparison")
        lines.append("Models:")
        for i, name in enumerate(self.models):
            lines.append(f"  {i + 1}: {name}")
        lines.append("")

        header = f"{'':8} {'npar':>6} {'AIC':>10} {'BIC':>10} {'logLik':>10} {'deviance':>10} {'Chisq':>8} {'Df':>4} {'Pr(>Chisq)':>12}"
        lines.append(header)

        for i in range(len(self.models)):
            name = f"Model {i + 1}"
            npar = self.df[i]
            aic = self.aic[i]
            bic = self.bic[i]
            loglik = self.loglik[i]
            dev = self.deviance[i]

            if self.chi_sq[i] is not None:
                chi_sq = f"{self.chi_sq[i]:8.4f}"
                chi_df = f"{self.chi_df[i]:4d}"
                p_val = self.p_value[i]
                if p_val is not None:
                    if p_val < 0.001:
                        p_str = f"{p_val:12.2e}"
                    else:
                        p_str = f"{p_val:12.4f}"
                else:
                    p_str = ""
            else:
                chi_sq = ""
                chi_df = ""
                p_str = ""

            lines.append(
                f"{name:8} {npar:6d} {aic:10.2f} {bic:10.2f} {loglik:10.2f} {dev:10.2f} {chi_sq:>8} {chi_df:>4} {p_str:>12}"
            )

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"AnovaResult(n_models={len(self.models)})"


def anova(
    *models: LmerResult | GlmerResult,
    refit: bool = True,
) -> AnovaResult:
    if len(models) < 2:
        raise ValueError("anova requires at least 2 models to compare")

    model_list = list(models)

    n_obs_list = [m.matrices.n_obs for m in model_list]
    if len(set(n_obs_list)) > 1:
        raise ValueError(
            f"Models have different numbers of observations: {n_obs_list}. "
            "Models must be fit to the same data for comparison."
        )

    from mixedlm.models.lmer import LmerResult

    is_lmer = [isinstance(m, LmerResult) for m in model_list]
    if refit and any(is_lmer):
        reml_flags = [m.REML for m in model_list if isinstance(m, LmerResult)]
        if any(reml_flags):
            import warnings

            warnings.warn(
                "Some models were fit with REML. For valid likelihood ratio tests, "
                "models should be fit with ML (REML=False). Consider refitting.",
                UserWarning,
                stacklevel=2,
            )

    model_names = []
    n_obs = []
    df_list = []
    aic_list = []
    bic_list = []
    loglik_list = []
    deviance_list = []

    for i, m in enumerate(model_list):
        model_names.append(str(m.formula))
        n_obs.append(m.matrices.n_obs)

        n_fixed = m.matrices.n_fixed
        n_theta = len(m.theta)
        if isinstance(m, LmerResult):
            n_params = n_fixed + n_theta + 1
        else:
            n_params = n_fixed + n_theta

        df_list.append(n_params)
        aic_list.append(m.AIC())
        bic_list.append(m.BIC())
        loglik_list.append(m.logLik())
        deviance_list.append(m.deviance)

    sorted_indices = sorted(range(len(df_list)), key=lambda i: df_list[i])
    model_names = [model_names[i] for i in sorted_indices]
    n_obs = [n_obs[i] for i in sorted_indices]
    df_list = [df_list[i] for i in sorted_indices]
    aic_list = [aic_list[i] for i in sorted_indices]
    bic_list = [bic_list[i] for i in sorted_indices]
    loglik_list = [loglik_list[i] for i in sorted_indices]
    deviance_list = [deviance_list[i] for i in sorted_indices]

    chi_sq: list[float | None] = [None]
    chi_df: list[int | None] = [None]
    p_value: list[float | None] = [None]

    for i in range(1, len(model_list)):
        ll_diff = 2 * (loglik_list[i] - loglik_list[i - 1])
        df_diff = df_list[i] - df_list[i - 1]

        if df_diff <= 0:
            chi_sq.append(None)
            chi_df.append(None)
            p_value.append(None)
        else:
            chi_sq.append(float(ll_diff))
            chi_df.append(df_diff)
            p_val = 1 - stats.chi2.cdf(ll_diff, df_diff)
            p_value.append(float(p_val))

    return AnovaResult(
        models=model_names,
        n_obs=n_obs,
        df=df_list,
        aic=aic_list,
        bic=bic_list,
        loglik=loglik_list,
        deviance=deviance_list,
        chi_sq=chi_sq,
        chi_df=chi_df,
        p_value=p_value,
    )
