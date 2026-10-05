from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from mixedlm._parallel import process_pool, resolve_n_jobs

if TYPE_CHECKING:
    import pandas as pd

    from mixedlm.families.base import Family
    from mixedlm.formula.terms import Formula
    from mixedlm.models.control import GlmerControl, LmerControl
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
            stacklevel=4,
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


def _refit_lmer(
    data: pd.DataFrame,
    weights: NDArray[np.floating] | None,
    offset: NDArray[np.floating] | None,
    control: LmerControl,
    start: NDArray[np.floating],
    formula: Formula,
) -> LmerResult:
    from mixedlm.models.lmer import LmerMod

    model = LmerMod(formula, data, REML=False, weights=weights, offset=offset, control=control)
    return model.fit(start=start)


def _refit_glmer(
    data: pd.DataFrame,
    family: Family,
    weights: NDArray[np.floating] | None,
    offset: NDArray[np.floating] | None,
    nAGQ: int,
    control: GlmerControl,
    start: NDArray[np.floating],
    formula: Formula,
) -> GlmerResult:
    from mixedlm.models.glmer import GlmerMod

    model = GlmerMod(formula, data, family=family, control=control, weights=weights, offset=offset)
    return model.fit(nAGQ=nAGQ, start=start)


def _drop1_worker(task: tuple[Any, ...]) -> Drop1WorkerResult:
    term, refit, formula, full_n_params, full_loglik, test = task
    try:
        reduced_model = refit(formula)
        reduced_n_params = reduced_model.npar()
        lrt, p_val = _likelihood_ratio(
            full_n_params, reduced_n_params, full_loglik, reduced_model.logLik().value, test
        )
        return (term, reduced_n_params, reduced_model.AIC(), lrt, p_val, None)
    except Exception as e:
        return (term, None, None, None, None, str(e))


def _drop1(
    model: LmerResult | GlmerResult,
    refit: Callable[[Formula], LmerResult | GlmerResult],
    test: str,
    n_jobs: int,
) -> Drop1Result:
    """Refit each valid single-term deletion, serially or in worker processes."""
    from mixedlm.formula.parser import update_formula

    droppable_terms = _droppable_fixed_terms(model)
    full_n_params = model.npar()
    full_loglik = model.logLik().value
    tasks = [
        (
            term,
            refit,
            update_formula(model.formula, f". ~ . - {term}"),
            full_n_params,
            full_loglik,
            test,
        )
        for term in droppable_terms
    ]
    workers = min(n_jobs, len(tasks))
    if workers <= 1:
        worker_results = [_drop1_worker(task) for task in tasks]
    else:
        with process_pool(workers) as executor:
            worker_results = list(executor.map(_drop1_worker, tasks))
    return _assemble_drop1_result(droppable_terms, worker_results, model.AIC(), full_n_params)


def drop1_lmer(
    model: LmerResult,
    data: pd.DataFrame,
    test: str = "Chisq",
    n_jobs: int = 1,
) -> Drop1Result:
    """Compare valid single-term deletions from a fitted linear mixed model.

    REML fits are automatically refitted with ML because likelihoods from
    different fixed-effects specifications are not comparable under REML.
    Reduced models reuse the fitted model's control settings.
    Terms contained in higher-order interactions are retained to respect the
    marginality principle. ``n_jobs`` worker processes, or -1 for all CPUs,
    refit deletions concurrently. Workers are started without forking, so
    scripts need an ``if __name__ == "__main__":`` guard.
    """
    test = _normalize_test(test)
    n_jobs = resolve_n_jobs(n_jobs)
    comparison_model = model.refitML() if model.REML else model
    matrices = comparison_model.matrices
    weights = matrices.weights if np.any(matrices.weights != 1.0) else None
    offset = matrices.offset if np.any(matrices.offset != 0.0) else None
    refit = partial(
        _refit_lmer, data, weights, offset, model._refit_control(), comparison_model.theta
    )
    return _drop1(comparison_model, refit, test, n_jobs)


def drop1_glmer(
    model: GlmerResult,
    data: pd.DataFrame,
    test: str = "Chisq",
    n_jobs: int = 1,
) -> Drop1Result:
    """Compare valid single-term deletions from a fitted generalized mixed model.

    Terms contained in higher-order interactions are retained to respect the
    marginality principle. ``n_jobs`` worker processes, or -1 for all CPUs,
    refit deletions concurrently. Workers are started without forking, so
    scripts need an ``if __name__ == "__main__":`` guard.
    """
    test = _normalize_test(test)
    n_jobs = resolve_n_jobs(n_jobs)
    # Rebuilding a count-response formula multiplies prior weights by trials.
    # Recover those priors so the reduced model applies trial counts once.
    prior_weights = (
        model.matrices.weights
        if model.matrices.trials is None
        else model.matrices.weights / model.matrices.trials
    )
    weights = prior_weights if np.any(prior_weights != 1.0) else None
    offset = model.matrices.offset if np.any(model.matrices.offset != 0.0) else None
    refit = partial(
        _refit_glmer,
        data,
        model.family,
        weights,
        offset,
        model.nAGQ,
        model._refit_control(),
        model.theta,
    )
    return _drop1(model, refit, test, n_jobs)
