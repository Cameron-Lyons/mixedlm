from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult


OPTIMIZERS = [
    "L-BFGS-B",
    "Nelder-Mead",
    "Powell",
    "BFGS",
    "TNC",
    "SLSQP",
]


@dataclass
class AllFitResult:
    fits: dict[str, LmerResult | GlmerResult | None]
    errors: dict[str, str]
    warnings: dict[str, list[str]]

    def __str__(self) -> str:
        lines = []
        lines.append("allFit summary:")
        lines.append("")

        lines.append(f"{'Optimizer':<15} {'Converged':>10} {'Deviance':>12} {'AIC':>12} {'Singular':>10}")
        lines.append("-" * 65)

        for opt_name, fit in self.fits.items():
            if fit is None:
                error_msg = self.errors.get(opt_name, "Unknown error")[:20]
                lines.append(f"{opt_name:<15} {'FAILED':>10} {'-':>12} {'-':>12} {'-':>10}")
            else:
                converged = "Yes" if fit.converged else "No"
                singular = "Yes" if fit.isSingular() else "No"
                lines.append(
                    f"{opt_name:<15} {converged:>10} {fit.deviance:>12.4f} {fit.AIC():>12.2f} {singular:>10}"
                )

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
        successful_fits = [f for f in self.fits.values() if f is not None]
        if len(successful_fits) < 2:
            return True

        deviances = [f.deviance for f in successful_fits]
        return (max(deviances) - min(deviances)) < tol


def allfit_lmer(
    model: LmerResult,
    data: pd.DataFrame,
    optimizers: list[str] | None = None,
    verbose: bool = False,
) -> AllFitResult:
    from mixedlm.models.lmer import LmerMod

    if optimizers is None:
        optimizers = OPTIMIZERS

    fits: dict[str, LmerResult | None] = {}
    errors: dict[str, str] = {}
    warnings: dict[str, list[str]] = {}

    for opt_name in optimizers:
        if verbose:
            print(f"Fitting with {opt_name}...")

        try:
            lmer_model = LmerMod(
                model.formula,
                data,
                REML=model.REML,
                weights=model.matrices.weights if np.any(model.matrices.weights != 1.0) else None,
                offset=model.matrices.offset if np.any(model.matrices.offset != 0.0) else None,
            )
            fit = lmer_model.fit(method=opt_name)
            fits[opt_name] = fit
            warnings[opt_name] = []

            if not fit.converged:
                warnings[opt_name].append("Did not converge")
            if fit.isSingular():
                warnings[opt_name].append("Singular fit")

        except Exception as e:
            fits[opt_name] = None
            errors[opt_name] = str(e)

    return AllFitResult(fits=fits, errors=errors, warnings=warnings)


def allfit_glmer(
    model: GlmerResult,
    data: pd.DataFrame,
    optimizers: list[str] | None = None,
    verbose: bool = False,
) -> AllFitResult:
    from mixedlm.models.glmer import GlmerMod

    if optimizers is None:
        optimizers = OPTIMIZERS

    fits: dict[str, GlmerResult | None] = {}
    errors: dict[str, str] = {}
    warnings: dict[str, list[str]] = {}

    for opt_name in optimizers:
        if verbose:
            print(f"Fitting with {opt_name}...")

        try:
            glmer_model = GlmerMod(
                model.formula,
                data,
                family=model.family,
                weights=model.matrices.weights if np.any(model.matrices.weights != 1.0) else None,
                offset=model.matrices.offset if np.any(model.matrices.offset != 0.0) else None,
            )
            fit = glmer_model.fit(method=opt_name, nAGQ=model.nAGQ)
            fits[opt_name] = fit
            warnings[opt_name] = []

            if not fit.converged:
                warnings[opt_name].append("Did not converge")
            if fit.isSingular():
                warnings[opt_name].append("Singular fit")

        except Exception as e:
            fits[opt_name] = None
            errors[opt_name] = str(e)

    return AllFitResult(fits=fits, errors=errors, warnings=warnings)
