from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from scipy import stats

if TYPE_CHECKING:
    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult


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

        header = (
            f"{'':8} {'npar':>6} {'AIC':>10} {'BIC':>10} {'logLik':>10} "
            f"{'deviance':>10} {'Chisq':>8} {'Df':>4} {'Pr(>Chisq)':>12}"
        )
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
                    p_str = f"{p_val:12.2e}" if p_val < 0.001 else f"{p_val:12.4f}"
                else:
                    p_str = ""
            else:
                chi_sq = ""
                chi_df = ""
                p_str = ""

            line = (
                f"{name:8} {npar:6d} {aic:10.2f} {bic:10.2f} {loglik:10.2f} "
                f"{dev:10.2f} {chi_sq:>8} {chi_df:>4} {p_str:>12}"
            )
            lines.append(line)

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

    model_data = []
    for m in model_list:
        n_fixed = m.matrices.n_fixed
        n_theta = len(m.theta)
        n_params = n_fixed + n_theta + 1 if isinstance(m, LmerResult) else n_fixed + n_theta

        model_data.append((
            str(m.formula),
            m.matrices.n_obs,
            n_params,
            m.AIC(),
            m.BIC(),
            m.logLik().value,
            m.deviance,
        ))

    model_data.sort(key=lambda x: x[2])

    model_names = [d[0] for d in model_data]
    n_obs = [d[1] for d in model_data]
    df_list = [d[2] for d in model_data]
    aic_list = [d[3] for d in model_data]
    bic_list = [d[4] for d in model_data]
    loglik_list = [d[5] for d in model_data]
    deviance_list = [d[6] for d in model_data]

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
