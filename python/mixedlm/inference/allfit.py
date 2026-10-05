from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mixedlm._parallel import process_pool, resolve_n_jobs

if TYPE_CHECKING:
    import pandas as pd

    from mixedlm.families.base import Family
    from mixedlm.formula.terms import Formula
    from mixedlm.models.control import GlmerControl, LmerControl
    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult


def _default_optimizers() -> list[str]:
    from mixedlm.estimation.optimizers import COMPATIBILITY_OPTIMIZERS, available_optimizers

    return [name for name in available_optimizers() if name not in COMPATIBILITY_OPTIMIZERS]


@dataclass
class AllFitResult:
    fits: dict[str, LmerResult | GlmerResult | None]
    errors: dict[str, str]
    warnings: dict[str, list[str]]

    def __str__(self) -> str:
        lines = []
        lines.append("allFit summary:")
        lines.append("")

        header = f"{'Optimizer':<15} {'Converged':>10} {'Deviance':>12} {'AIC':>12}"
        header += f" {'Singular':>10}"
        lines.append(header)
        lines.append("-" * 65)

        for opt_name, fit in self.fits.items():
            if fit is None:
                lines.append(f"{opt_name:<15} {'FAILED':>10} {'-':>12} {'-':>12} {'-':>10}")
            else:
                converged = "Yes" if fit.converged else "No"
                singular = "Yes" if fit.isSingular() else "No"
                row = f"{opt_name:<15} {converged:>10} {fit.deviance:>12.4f}"
                row += f" {fit.AIC():>12.2f} {singular:>10}"
                lines.append(row)

        if self.errors:
            lines.append("")
            lines.append("Errors:")
            for opt_name, error in self.errors.items():
                lines.append(f"  {opt_name}: {error}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        n_success = sum(1 for f in self.fits.values() if f is not None)
        n_total = len(self.fits)
        return f"AllFitResult({n_success}/{n_total} successful)"

    @property
    def results(self) -> dict[str, LmerResult | GlmerResult | Exception]:
        """Return fits and failures in the legacy combined mapping."""
        return {
            name: fit
            if fit is not None
            else RuntimeError(self.errors.get(name, "Fit failed without an error message"))
            for name, fit in self.fits.items()
        }

    @property
    def summary(self) -> pd.DataFrame:
        """Return a tabular optimizer comparison."""
        import pandas as pd

        rows: list[dict[str, Any]] = []
        for name, fit in self.fits.items():
            if fit is None:
                rows.append(
                    {
                        "optimizer": name,
                        "converged": False,
                        "deviance": np.nan,
                        "iterations": np.nan,
                        "gradient_norm": np.nan,
                        "at_boundary": np.nan,
                        "error": self.errors.get(name),
                    }
                )
            else:
                gradient_norm = getattr(fit, "gradient_norm", None)
                rows.append(
                    {
                        "optimizer": name,
                        "converged": fit.converged,
                        "deviance": fit.deviance,
                        "iterations": fit.n_iter,
                        "gradient_norm": gradient_norm if gradient_norm is not None else np.nan,
                        "at_boundary": getattr(fit, "at_boundary", np.nan),
                        "error": None,
                    }
                )
        return pd.DataFrame(rows)

    @property
    def best_optimizer(self) -> str:
        """Return the optimizer with the lowest deviance."""
        successful = {name: fit for name, fit in self.fits.items() if fit is not None}
        if not successful:
            return next(iter(self.fits), "")
        return min(successful, key=lambda name: successful[name].deviance)

    def fixef_table(self) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for opt_name, fit in self.fits.items():
            if fit is not None:
                result[opt_name] = fit.fixef()
        return result

    def theta_table(self) -> dict[str, list[float]]:
        result: dict[str, list[float]] = {}
        for opt_name, fit in self.fits.items():
            if fit is not None:
                result[opt_name] = list(fit.theta)
        return result

    def best_fit(self, criterion: str = "deviance") -> LmerResult | GlmerResult | None:
        successful_fits = {k: v for k, v in self.fits.items() if v is not None}
        if not successful_fits:
            return None

        if criterion == "deviance":
            return min(successful_fits.values(), key=lambda x: x.deviance)
        elif criterion == "AIC":
            return min(successful_fits.values(), key=lambda x: x.AIC())
        elif criterion == "BIC":
            return min(successful_fits.values(), key=lambda x: x.BIC())
        else:
            raise ValueError(f"Unknown criterion: {criterion}. Use 'deviance', 'AIC', or 'BIC'.")

    def is_consistent(self, tol: float = 1e-3) -> bool:
        """Return whether every converged fit reaches the same deviance within ``tol``."""
        fits = [fit for fit in self.fits.values() if fit is not None and fit.converged]
        deviances = [fit.deviance for fit in fits]
        return len(deviances) < 2 or bool(max(deviances) - min(deviances) < tol)


def _fit_with_optimizer(
    fit: Callable[[str], LmerResult | GlmerResult], optimizer: str
) -> tuple[LmerResult | GlmerResult | None, str | None, list[str]]:
    try:
        result = fit(optimizer)
        messages = []
        if not result.converged:
            messages.append("Did not converge")
        if result.isSingular():
            messages.append("Singular fit")
    except Exception as error:
        return None, str(error), []
    return result, None, messages


def _run_allfit(
    fit: Callable[[str], LmerResult | GlmerResult],
    optimizers: list[str],
    n_jobs: int,
    verbose: bool,
) -> AllFitResult:
    """Fit once per optimizer, serially or in worker processes, with the same worker."""
    workers = resolve_n_jobs(n_jobs, max_tasks=len(optimizers))
    attempt = partial(_fit_with_optimizer, fit)
    fits: dict[str, LmerResult | GlmerResult | None] = {}
    errors: dict[str, str] = {}
    warnings: dict[str, list[str]] = {}
    with process_pool(workers) if workers > 1 else nullcontext() as executor:
        run = map if executor is None else executor.map
        for name, (result, error, messages) in zip(
            optimizers, run(attempt, optimizers), strict=True
        ):
            fits[name] = result
            warnings[name] = messages
            if error is not None:
                errors[name] = error
            if verbose:
                summary = f"ERROR: {error}" if result is None else f"deviance={result.deviance:.4f}"
                print("; ".join([f"{name}: {summary}", *messages]))
    return AllFitResult(fits=fits, errors=errors, warnings=warnings)


def _refit_lmer(
    formula: Formula,
    data: pd.DataFrame,
    REML: bool,
    weights: NDArray[np.floating] | None,
    offset: NDArray[np.floating] | None,
    control: LmerControl,
    optimizer: str,
) -> LmerResult:
    from mixedlm.models.lmer import LmerMod

    model = LmerMod(formula, data, REML=REML, weights=weights, offset=offset, control=control)
    return model.fit(method=optimizer)


def _refit_glmer(
    formula: Formula,
    data: pd.DataFrame,
    family: Family,
    weights: NDArray[np.floating] | None,
    offset: NDArray[np.floating] | None,
    nAGQ: int,
    control: GlmerControl,
    optimizer: str,
) -> GlmerResult:
    from mixedlm.models.glmer import GlmerMod

    model = GlmerMod(formula, data, family=family, control=control, weights=weights, offset=offset)
    return model.fit(method=optimizer, nAGQ=nAGQ)


def allfit_lmer(
    model: LmerResult,
    data: pd.DataFrame,
    optimizers: list[str] | None = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> AllFitResult:
    """Refit an LMM with each optimizer, keeping its other control settings.

    ``n_jobs`` worker processes, or -1 for all CPUs, run the refits. Workers are
    started without forking, so scripts need an ``if __name__ == "__main__":``
    guard.
    """
    if optimizers is None:
        optimizers = _default_optimizers()
    weights = model.matrices.weights if np.any(model.matrices.weights != 1.0) else None
    offset = model.matrices.offset if np.any(model.matrices.offset != 0.0) else None
    fit = partial(
        _refit_lmer, model.formula, data, model.REML, weights, offset, model._refit_control()
    )
    return _run_allfit(fit, optimizers, n_jobs, verbose)


def allfit_glmer(
    model: GlmerResult,
    data: pd.DataFrame,
    optimizers: list[str] | None = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> AllFitResult:
    """Refit a GLMM with each optimizer, keeping its quadrature and inner controls.

    ``n_jobs`` worker processes, or -1 for all CPUs, run the refits. Workers are
    started without forking, so scripts need an ``if __name__ == "__main__":``
    guard.
    """
    if optimizers is None:
        optimizers = _default_optimizers()
    # Count-response formulas apply trial counts when rebuilding the matrices.
    prior_weights = (
        model.matrices.weights
        if model.matrices.trials is None
        else model.matrices.weights / model.matrices.trials
    )
    weights = prior_weights if np.any(prior_weights != 1.0) else None
    offset = model.matrices.offset if np.any(model.matrices.offset != 0.0) else None
    fit = partial(
        _refit_glmer,
        model.formula,
        data,
        model.family,
        weights,
        offset,
        model.nAGQ,
        model._refit_control(),
    )
    return _run_allfit(fit, optimizers, n_jobs, verbose)
