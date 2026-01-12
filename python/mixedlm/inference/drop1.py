from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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
                p_str = f"{p_val:12.2e}" if p_val < 0.001 else f"{p_val:12.4f}"
            else:
                lrt_str = " " * 10
                p_str = " " * 12

            lines.append(f"- {term:<18} {df:>4} {aic:>10.2f} {lrt_str} {p_str}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Drop1Result(n_terms={len(self.terms)})"


def _drop1_lmer_worker(
    args: tuple[Any, ...],
) -> tuple[str, int | None, float | None, float | None, float | None, str | None]:
    from mixedlm.estimation.reml import _count_theta
    from mixedlm.models.lmer import LmerMod

    term, formula, data, REML, weights, offset, full_n_params, full_loglik, test = args

    try:
        lmer_model = LmerMod(
            formula,
            data,
            REML=REML,
            weights=weights,
            offset=offset,
        )
        reduced_model = lmer_model.fit()

        reduced_n_theta = _count_theta(reduced_model.matrices.random_structures)
        reduced_n_params = reduced_model.matrices.n_fixed + reduced_n_theta + 1
        reduced_aic = reduced_model.AIC()
        reduced_loglik = reduced_model.logLik().value

        lrt: float | None = None
        p_val: float | None = None

        if test == "Chisq":
            df_diff = full_n_params - reduced_n_params
            if df_diff > 0:
                lrt = 2 * (full_loglik - reduced_loglik)
                p_val = 1 - stats.chi2.cdf(lrt, df_diff)

        return (term, reduced_n_params, reduced_aic, lrt, p_val, None)
    except Exception as e:
        return (term, None, None, None, None, str(e))


def _drop1_glmer_worker(
    args: tuple[Any, ...],
) -> tuple[str, int | None, float | None, float | None, float | None, str | None]:
    from mixedlm.estimation.laplace import _count_theta
    from mixedlm.models.glmer import GlmerMod

    term, formula, data, family, weights, offset, nAGQ, full_n_params, full_loglik, test = args

    try:
        glmer_model = GlmerMod(
            formula,
            data,
            family=family,
            weights=weights,
            offset=offset,
        )
        reduced_model = glmer_model.fit(nAGQ=nAGQ)

        reduced_n_theta = _count_theta(reduced_model.matrices.random_structures)
        reduced_n_params = reduced_model.matrices.n_fixed + reduced_n_theta
        reduced_aic = reduced_model.AIC()
        reduced_loglik = reduced_model.logLik().value

        lrt: float | None = None
        p_val: float | None = None

        if test == "Chisq":
            df_diff = full_n_params - reduced_n_params
            if df_diff > 0:
                lrt = 2 * (full_loglik - reduced_loglik)
                p_val = 1 - stats.chi2.cdf(lrt, df_diff)

        return (term, reduced_n_params, reduced_aic, lrt, p_val, None)
    except Exception as e:
        return (term, None, None, None, None, str(e))


def drop1_lmer(
    model: LmerResult,
    data: pd.DataFrame,
    test: str = "Chisq",
    n_jobs: int = 1,
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

    weights = model.matrices.weights if np.any(model.matrices.weights != 1.0) else None
    offset = model.matrices.offset if np.any(model.matrices.offset != 0.0) else None

    if n_jobs == 1:
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
    else:
        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1

        from mixedlm.formula.parser import update_formula

        tasks = []
        for term in droppable_terms:
            reduced_formula = update_formula(model.formula, f". ~ . - {term}")
            tasks.append((
                term,
                reduced_formula,
                data,
                model.REML,
                weights,
                offset,
                full_n_params,
                full_loglik,
                test,
            ))

        results: dict[str, tuple[int | None, float | None, float | None, float | None]] = {}

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(_drop1_lmer_worker, task): task[0] for task in tasks}

            for future in as_completed(futures):
                term, n_params, aic, lrt, p_val, error = future.result()
                if error is None and n_params is not None:
                    results[term] = (n_params, aic, lrt, p_val)

        terms = []
        df_list = []
        aic_list = []
        lrt_list = []
        p_value_list = []

        for term in droppable_terms:
            if term in results:
                n_params, aic, lrt, p_val = results[term]
                terms.append(term)
                df_list.append(n_params)
                aic_list.append(aic)
                lrt_list.append(lrt)
                p_value_list.append(p_val)

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
    n_jobs: int = 1,
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

    weights = model.matrices.weights if np.any(model.matrices.weights != 1.0) else None
    offset = model.matrices.offset if np.any(model.matrices.offset != 0.0) else None

    if n_jobs == 1:
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
    else:
        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1

        from mixedlm.formula.parser import update_formula

        tasks = []
        for term in droppable_terms:
            reduced_formula = update_formula(model.formula, f". ~ . - {term}")
            tasks.append((
                term,
                reduced_formula,
                data,
                model.family,
                weights,
                offset,
                model.nAGQ,
                full_n_params,
                full_loglik,
                test,
            ))

        results: dict[str, tuple[int | None, float | None, float | None, float | None]] = {}

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(_drop1_glmer_worker, task): task[0] for task in tasks}

            for future in as_completed(futures):
                term, n_params, aic, lrt, p_val, error = future.result()
                if error is None and n_params is not None:
                    results[term] = (n_params, aic, lrt, p_val)

        terms = []
        df_list = []
        aic_list = []
        lrt_list = []
        p_value_list = []

        for term in droppable_terms:
            if term in results:
                n_params, aic, lrt, p_val = results[term]
                terms.append(term)
                df_list.append(n_params)
                aic_list.append(aic)
                lrt_list.append(lrt)
                p_value_list.append(p_val)

    return Drop1Result(
        terms=terms,
        df=df_list,
        aic=aic_list,
        lrt=lrt_list,
        p_value=p_value_list,
        full_model_aic=full_aic,
        full_model_df=full_n_params,
    )
