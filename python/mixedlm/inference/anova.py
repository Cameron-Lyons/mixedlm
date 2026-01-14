from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import stats

if TYPE_CHECKING:
    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult


@dataclass
class AnovaType3Result:
    """Result of Type III ANOVA for a single model.

    Type III tests are marginal tests that evaluate each term's contribution
    after adjusting for all other terms. For mixed models, F-tests use
    Satterthwaite or Kenward-Roger denominator degrees of freedom.
    """

    terms: list[str]
    sum_sq: NDArray[np.floating]
    mean_sq: NDArray[np.floating]
    num_df: NDArray[np.int64]
    den_df: NDArray[np.floating]
    f_value: NDArray[np.floating]
    p_value: NDArray[np.floating]
    ddf_method: str

    def __str__(self) -> str:
        lines = []
        lines.append(f"Type III Analysis of Variance Table with {self.ddf_method} DF")
        lines.append("")
        lines.append(
            f"{'Term':<20} {'Sum Sq':>12} {'Mean Sq':>12} {'NumDF':>6} "
            f"{'DenDF':>8} {'F value':>10} {'Pr(>F)':>12}"
        )
        lines.append("-" * 82)

        for i, term in enumerate(self.terms):
            p_val = self.p_value[i]
            sig = _get_signif_code(p_val)
            p_str = f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.4f}"
            lines.append(
                f"{term:<20} {self.sum_sq[i]:>12.4f} {self.mean_sq[i]:>12.4f} "
                f"{self.num_df[i]:>6d} {self.den_df[i]:>8.2f} {self.f_value[i]:>10.4f} "
                f"{p_str:>12} {sig}"
            )

        lines.append("---")
        lines.append("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"AnovaType3Result(terms={self.terms})"


def _get_signif_code(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.1:
        return "."
    return ""


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

        model_data.append(
            (
                str(m.formula),
                m.matrices.n_obs,
                n_params,
                m.AIC(),
                m.BIC(),
                m.logLik().value,
                m.deviance,
            )
        )

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


def anova_type3(
    model: LmerResult,
    ddf_method: str = "Satterthwaite",
) -> AnovaType3Result:
    """Compute Type III ANOVA table for a linear mixed model.

    Type III tests evaluate each fixed effect term's contribution after
    adjusting for all other terms in the model. Uses F-tests with
    Satterthwaite or Kenward-Roger denominator degrees of freedom.

    Parameters
    ----------
    model : LmerResult
        A fitted linear mixed model.
    ddf_method : str, default "Satterthwaite"
        Method for computing denominator degrees of freedom.
        Options: "Satterthwaite", "Kenward-Roger".

    Returns
    -------
    AnovaType3Result
        Type III ANOVA table with F-tests for each fixed effect term.

    Notes
    -----
    This function tests terms (factors/covariates) rather than individual
    coefficients. For factors with multiple levels, it computes a joint
    F-test across all associated coefficients.

    For continuous covariates, the test is equivalent to the t-test
    from summary() but expressed as an F-test (F = t^2).

    Examples
    --------
    >>> result = lmer("Reaction ~ Days + (1|Subject)", sleepstudy)
    >>> anova_type3(result)
    Type III Analysis of Variance Table with Satterthwaite DF
    Term                       Sum Sq       Mean Sq  NumDF    DenDF    F value       Pr(>F)
    Days                      30031.2       30031.2      1    17.00     45.853     3.26e-06 ***
    """
    from scipy import linalg

    from mixedlm.inference.ddf import kenward_roger_df, satterthwaite_df
    from mixedlm.models.lmer import LmerResult

    if not isinstance(model, LmerResult):
        raise TypeError("anova_type3 currently only supports LmerResult objects")

    if ddf_method == "Satterthwaite":
        ddf_result = satterthwaite_df(model)
    elif ddf_method == "Kenward-Roger":
        ddf_result = kenward_roger_df(model)
    else:
        raise ValueError(
            f"Unknown ddf_method: {ddf_method}. Use 'Satterthwaite' or 'Kenward-Roger'."
        )

    vcov = model.vcov()
    beta = model.beta
    sigma2 = model.sigma**2

    term_groups = _get_term_groups(model)

    terms = []
    sum_sq_list = []
    mean_sq_list = []
    num_df_list = []
    den_df_list = []
    f_value_list = []
    p_value_list = []

    for term_name, indices in term_groups.items():
        if term_name == "(Intercept)":
            continue

        indices_arr = np.array(indices)
        num_df = len(indices)

        beta_term = beta[indices_arr]
        vcov_term = vcov[np.ix_(indices_arr, indices_arr)]

        try:
            vcov_term_inv = linalg.inv(vcov_term)
        except linalg.LinAlgError:
            vcov_term_inv = linalg.pinv(vcov_term)

        f_stat = float(beta_term @ vcov_term_inv @ beta_term) / num_df

        avg_den_df = float(np.mean(ddf_result.df[indices_arr]))

        p_val = 1 - stats.f.cdf(f_stat, num_df, avg_den_df)

        ss = f_stat * num_df * sigma2
        ms = ss / num_df

        terms.append(term_name)
        sum_sq_list.append(ss)
        mean_sq_list.append(ms)
        num_df_list.append(num_df)
        den_df_list.append(avg_den_df)
        f_value_list.append(f_stat)
        p_value_list.append(p_val)

    return AnovaType3Result(
        terms=terms,
        sum_sq=np.array(sum_sq_list),
        mean_sq=np.array(mean_sq_list),
        num_df=np.array(num_df_list, dtype=np.int64),
        den_df=np.array(den_df_list),
        f_value=np.array(f_value_list),
        p_value=np.array(p_value_list),
        ddf_method=ddf_method,
    )


def _get_term_groups(model: LmerResult) -> dict[str, list[int]]:
    """Group coefficient indices by their originating term."""
    fixed_names = list(model.matrices.fixed_names)
    groups: dict[str, list[int]] = {}

    for i, name in enumerate(fixed_names):
        if name == "(Intercept)":
            term = "(Intercept)"
        elif ":" in name:
            parts = name.split(":")
            base_parts = []
            for p in parts:
                if "[" in p:
                    base_parts.append(p.split("[")[0])
                else:
                    base_parts.append(p)
            term = ":".join(sorted(base_parts))
        elif "[" in name:
            term = name.split("[")[0]
        else:
            term = name

        if term not in groups:
            groups[term] = []
        groups[term].append(i)

    return groups
