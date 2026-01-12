from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    import pandas as pd

    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult


@dataclass
class Drop1Result:
    terms: list[str]
    df: list[int]
    aic: list[float]
    lrt: list[float | None]
    p_value: list[float | None]
    full_model_aic: float
    full_model_df: int

    def __str__(self) -> str:
        lines = []
        lines.append("Single term deletions")
        lines.append("")
        lines.append("Model:")
        lines.append(f"  Full model AIC: {self.full_model_aic:.2f}")
        lines.append("")

        header = f"{'Term':<20} {'Df':>4} {'AIC':>10} {'LRT':>10} {'Pr(>Chi)':>12}"
        lines.append(header)

        lines.append(f"{'<none>':<20} {self.full_model_df:>4} {self.full_model_aic:>10.2f}")

        for i in range(len(self.terms)):
            term = self.terms[i]
            df = self.df[i]
            aic = self.aic[i]
            lrt = self.lrt[i]
            p_val = self.p_value[i]

            if lrt is not None and p_val is not None:
                lrt_str = f"{lrt:10.4f}"
                if p_val < 0.001:
                    p_str = f"{p_val:12.2e}"
                else:
                    p_str = f"{p_val:12.4f}"
            else:
                lrt_str = " " * 10
                p_str = " " * 12

            lines.append(f"- {term:<18} {df:>4} {aic:>10.2f} {lrt_str} {p_str}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Drop1Result(n_terms={len(self.terms)})"


def drop1_lmer(
    model: LmerResult,
    data: pd.DataFrame,
    test: str = "Chisq",
) -> Drop1Result:
    from mixedlm.estimation.reml import _count_theta
    from mixedlm.formula.terms import InteractionTerm, VariableTerm

    droppable_terms: list[str] = []
    for term in model.formula.fixed.terms:
        if isinstance(term, VariableTerm):
            droppable_terms.append(term.name)
        elif isinstance(term, InteractionTerm):
            droppable_terms.append(":".join(term.variables))

    n_theta = _count_theta(model.matrices.random_structures)
    full_n_params = model.matrices.n_fixed + n_theta + 1
    full_aic = model.AIC()
    full_loglik = model.logLik().value

    terms: list[str] = []
    df_list: list[int] = []
    aic_list: list[float] = []
    lrt_list: list[float | None] = []
    p_value_list: list[float | None] = []

    for term in droppable_terms:
        try:
            reduced_model = model.update(f". ~ . - {term}", data=data)

            reduced_n_theta = _count_theta(reduced_model.matrices.random_structures)
            reduced_n_params = reduced_model.matrices.n_fixed + reduced_n_theta + 1
            reduced_aic = reduced_model.AIC()
            reduced_loglik = reduced_model.logLik().value

            terms.append(term)
            df_list.append(reduced_n_params)
            aic_list.append(reduced_aic)

            if test == "Chisq":
                df_diff = full_n_params - reduced_n_params
                if df_diff > 0:
                    lrt = 2 * (full_loglik - reduced_loglik)
                    p_val = 1 - stats.chi2.cdf(lrt, df_diff)
                    lrt_list.append(float(lrt))
                    p_value_list.append(float(p_val))
                else:
                    lrt_list.append(None)
                    p_value_list.append(None)
            else:
                lrt_list.append(None)
                p_value_list.append(None)

        except Exception:
            continue

    return Drop1Result(
        terms=terms,
        df=df_list,
        aic=aic_list,
        lrt=lrt_list,
        p_value=p_value_list,
        full_model_aic=full_aic,
        full_model_df=full_n_params,
    )


def drop1_glmer(
    model: GlmerResult,
    data: pd.DataFrame,
    test: str = "Chisq",
) -> Drop1Result:
    from mixedlm.estimation.laplace import _count_theta
    from mixedlm.formula.terms import InteractionTerm, VariableTerm

    droppable_terms: list[str] = []
    for term in model.formula.fixed.terms:
        if isinstance(term, VariableTerm):
            droppable_terms.append(term.name)
        elif isinstance(term, InteractionTerm):
            droppable_terms.append(":".join(term.variables))

    n_theta = _count_theta(model.matrices.random_structures)
    full_n_params = model.matrices.n_fixed + n_theta
    full_aic = model.AIC()
    full_loglik = model.logLik().value

    terms: list[str] = []
    df_list: list[int] = []
    aic_list: list[float] = []
    lrt_list: list[float | None] = []
    p_value_list: list[float | None] = []

    for term in droppable_terms:
        try:
            reduced_model = model.update(f". ~ . - {term}", data=data)

            reduced_n_theta = _count_theta(reduced_model.matrices.random_structures)
            reduced_n_params = reduced_model.matrices.n_fixed + reduced_n_theta
            reduced_aic = reduced_model.AIC()
            reduced_loglik = reduced_model.logLik().value

            terms.append(term)
            df_list.append(reduced_n_params)
            aic_list.append(reduced_aic)

            if test == "Chisq":
                df_diff = full_n_params - reduced_n_params
                if df_diff > 0:
                    lrt = 2 * (full_loglik - reduced_loglik)
                    p_val = 1 - stats.chi2.cdf(lrt, df_diff)
                    lrt_list.append(float(lrt))
                    p_value_list.append(float(p_val))
                else:
                    lrt_list.append(None)
                    p_value_list.append(None)
            else:
                lrt_list.append(None)
                p_value_list.append(None)

        except Exception:
            continue

    return Drop1Result(
        terms=terms,
        df=df_list,
        aic=aic_list,
        lrt=lrt_list,
        p_value=p_value_list,
        full_model_aic=full_aic,
        full_model_df=full_n_params,
    )
