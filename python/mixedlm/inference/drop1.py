from __future__ import annotations

import os
import warnings
from collections.abc import Callable
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


Drop1WorkerResult = tuple[
    str,
    int | None,
    float | None,
    float | None,
    float | None,
    str | None,
]


def _normalize_test(test: str) -> str:
    normalized = test.lower()
    if normalized == "chisq":
        return "Chisq"
    if normalized == "none":
        return "none"
    raise ValueError("test must be 'Chisq' or 'none'")


def _resolve_worker_count(n_jobs: int) -> int:
    if n_jobs == -1:
        return os.cpu_count() or 1
    if n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer or -1")
    return n_jobs


def _droppable_fixed_terms(model: LmerResult | GlmerResult) -> list[str]:
    """Return fixed terms whose deletion respects the marginality principle."""
    from mixedlm.formula.terms import (
        InteractionTerm,
        PowerTerm,
        VariableTerm,
        format_term,
    )

    FactorKey = tuple[str, int | None]

    def factor_key(factor: str | PowerTerm) -> FactorKey:
        if isinstance(factor, PowerTerm):
            return (factor.name, factor.exponent)
        return (factor, None)

    candidates: list[tuple[str, frozenset[FactorKey]]] = []
    for term in model.formula.fixed.terms:
        if isinstance(term, VariableTerm):
            candidates.append((format_term(term), frozenset((factor_key(term.name),))))
        elif isinstance(term, PowerTerm):
            candidates.append((format_term(term), frozenset((factor_key(term),))))
        elif isinstance(term, InteractionTerm):
            candidates.append(
                (
                    format_term(term),
                    frozenset(factor_key(factor) for factor in term.variables),
                )
            )

    return [
        label
        for index, (label, variables) in enumerate(candidates)
        if not any(
            variables < other_variables
            for other_index, (_, other_variables) in enumerate(candidates)
            if other_index != index
        )
    ]


def _likelihood_ratio(
    full_n_params: int,
    reduced_n_params: int,
    full_loglik: float,
    reduced_loglik: float,
    test: str,
) -> tuple[float | None, float | None]:
    if test != "Chisq":
        return None, None

    df_diff = full_n_params - reduced_n_params
    if df_diff <= 0:
        return None, None

    lrt = max(0.0, 2.0 * (full_loglik - reduced_loglik))
    return lrt, float(stats.chi2.sf(lrt, df_diff))


def _run_drop1_tasks(
    worker: Callable[[tuple[Any, ...]], Drop1WorkerResult],
    tasks: list[tuple[Any, ...]],
    n_jobs: int,
) -> list[Drop1WorkerResult]:
    if not tasks:
        return []

    worker_count = min(n_jobs, len(tasks))
    if worker_count == 1:
        return [worker(task) for task in tasks]

    results: list[Drop1WorkerResult] = []
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(worker, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def _assemble_drop1_result(
    droppable_terms: list[str],
    worker_results: list[Drop1WorkerResult],
    full_aic: float,
    full_n_params: int,
) -> Drop1Result:
    failures = [(term, error) for term, *_, error in worker_results if error is not None]
    if failures:
        details = "; ".join(f"{term}: {error}" for term, error in failures)
        warnings.warn(
            f"Single-term deletion failed for {len(failures)} term(s): {details}",
            RuntimeWarning,
            stacklevel=3,
        )

    successful = {
        term: (n_params, aic, lrt, p_value)
        for term, n_params, aic, lrt, p_value, error in worker_results
        if error is None and n_params is not None and aic is not None
    }

    terms = [term for term in droppable_terms if term in successful]
    return Drop1Result(
        terms=terms,
        df=[successful[term][0] for term in terms],
        aic=[successful[term][1] for term in terms],
        lrt=[successful[term][2] for term in terms],
        p_value=[successful[term][3] for term in terms],
        full_model_aic=full_aic,
        full_model_df=full_n_params,
    )


def _drop1_lmer_worker(
    args: tuple[Any, ...],
) -> Drop1WorkerResult:
    from mixedlm.estimation.reml import _count_theta
    from mixedlm.models.lmer import LmerMod

    term, formula, data, weights, offset, start, full_n_params, full_loglik, test = args

    try:
        lmer_model = LmerMod(
            formula,
            data,
            REML=False,
            weights=weights,
            offset=offset,
        )
        reduced_model = lmer_model.fit(start=start)

        reduced_n_theta = _count_theta(reduced_model.matrices.random_structures)
        reduced_n_params = reduced_model.matrices.n_fixed + reduced_n_theta + 1
        reduced_aic = reduced_model.AIC()
        reduced_loglik = reduced_model.logLik().value

        lrt, p_val = _likelihood_ratio(
            full_n_params,
            reduced_n_params,
            full_loglik,
            reduced_loglik,
            test,
        )

        return (term, reduced_n_params, reduced_aic, lrt, p_val, None)
    except Exception as e:
        return (term, None, None, None, None, str(e))


def _drop1_glmer_worker(
    args: tuple[Any, ...],
) -> Drop1WorkerResult:
    from mixedlm.estimation.laplace import _count_theta
    from mixedlm.models.glmer import GlmerMod

    (
        term,
        formula,
        data,
        family,
        weights,
        offset,
        nAGQ,
        start,
        full_n_params,
        full_loglik,
        test,
    ) = args

    try:
        glmer_model = GlmerMod(
            formula,
            data,
            family=family,
            weights=weights,
            offset=offset,
        )
        reduced_model = glmer_model.fit(nAGQ=nAGQ, start=start)

        reduced_n_theta = _count_theta(reduced_model.matrices.random_structures)
        reduced_n_params = reduced_model.matrices.n_fixed + reduced_n_theta
        reduced_aic = reduced_model.AIC()
        reduced_loglik = reduced_model.logLik().value

        lrt, p_val = _likelihood_ratio(
            full_n_params,
            reduced_n_params,
            full_loglik,
            reduced_loglik,
            test,
        )

        return (term, reduced_n_params, reduced_aic, lrt, p_val, None)
    except Exception as e:
        return (term, None, None, None, None, str(e))


def drop1_lmer(
    model: LmerResult,
    data: pd.DataFrame,
    test: str = "Chisq",
    n_jobs: int = 1,
) -> Drop1Result:
    """Compare valid single-term deletions from a fitted linear mixed model.

    REML fits are automatically refitted with ML because likelihoods from
    different fixed-effects specifications are not comparable under REML.
    Terms contained in higher-order interactions are retained to respect the
    marginality principle.
    """
    from mixedlm.estimation.reml import _count_theta
    from mixedlm.formula.parser import update_formula

    test = _normalize_test(test)
    n_jobs = _resolve_worker_count(n_jobs)
    comparison_model = model.refitML() if model.REML else model
    droppable_terms = _droppable_fixed_terms(comparison_model)

    n_theta = _count_theta(comparison_model.matrices.random_structures)
    full_n_params = comparison_model.matrices.n_fixed + n_theta + 1
    full_aic = comparison_model.AIC()
    full_loglik = comparison_model.logLik().value
    weights = (
        comparison_model.matrices.weights
        if np.any(comparison_model.matrices.weights != 1.0)
        else None
    )
    offset = (
        comparison_model.matrices.offset
        if np.any(comparison_model.matrices.offset != 0.0)
        else None
    )

    tasks = [
        (
            term,
            update_formula(comparison_model.formula, f". ~ . - {term}"),
            data,
            weights,
            offset,
            comparison_model.theta,
            full_n_params,
            full_loglik,
            test,
        )
        for term in droppable_terms
    ]
    worker_results = _run_drop1_tasks(_drop1_lmer_worker, tasks, n_jobs)
    return _assemble_drop1_result(
        droppable_terms,
        worker_results,
        full_aic,
        full_n_params,
    )


def drop1_glmer(
    model: GlmerResult,
    data: pd.DataFrame,
    test: str = "Chisq",
    n_jobs: int = 1,
) -> Drop1Result:
    """Compare valid single-term deletions from a fitted generalized mixed model.

    Terms contained in higher-order interactions are retained to respect the
    marginality principle.
    """
    from mixedlm.estimation.laplace import _count_theta
    from mixedlm.formula.parser import update_formula

    test = _normalize_test(test)
    n_jobs = _resolve_worker_count(n_jobs)
    droppable_terms = _droppable_fixed_terms(model)
    n_theta = _count_theta(model.matrices.random_structures)
    full_n_params = model.matrices.n_fixed + n_theta
    full_aic = model.AIC()
    full_loglik = model.logLik().value
    weights = model.matrices.weights if np.any(model.matrices.weights != 1.0) else None
    offset = model.matrices.offset if np.any(model.matrices.offset != 0.0) else None

    tasks = [
        (
            term,
            update_formula(model.formula, f". ~ . - {term}"),
            data,
            model.family,
            weights,
            offset,
            model.nAGQ,
            model.theta,
            full_n_params,
            full_loglik,
            test,
        )
        for term in droppable_terms
    ]
    worker_results = _run_drop1_tasks(_drop1_glmer_worker, tasks, n_jobs)
    return _assemble_drop1_result(
        droppable_terms,
        worker_results,
        full_aic,
        full_n_params,
    )
