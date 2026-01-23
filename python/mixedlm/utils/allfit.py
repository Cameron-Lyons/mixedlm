from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from mixedlm.models.lmer import LmerResult


@dataclass
class AllFitResult:
    """Results from fitting with multiple optimizers.

    Attributes
    ----------
    results : dict[str, LmerResult | Exception]
        Dictionary mapping optimizer name to either a successful result or an exception.
    summary : pd.DataFrame
        Summary table comparing optimizer performance.
    best_optimizer : str
        Name of the optimizer with the lowest deviance.
    """

    results: dict[str, LmerResult | Exception]
    summary: pd.DataFrame
    best_optimizer: str

    def __str__(self) -> str:
        lines = ["allFit() summary:"]
        lines.append("=" * 80)
        lines.append(str(self.summary))
        lines.append("")
        lines.append(f"Best optimizer: {self.best_optimizer}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        n_success = sum(1 for r in self.results.values() if not isinstance(r, Exception))
        n_total = len(self.results)
        return f"AllFitResult({n_success}/{n_total} successful, best={self.best_optimizer})"


def allFit(
    formula: str,
    data: pd.DataFrame,
    optimizers: list[str] | None = None,
    REML: bool = True,
    verbose: int = 0,
    **kwargs,
) -> AllFitResult:
    """Fit a model with multiple optimizers and compare results.

    This function fits the same model using different optimization algorithms
    and compares the results. This is useful for:
    - Verifying convergence (all optimizers should reach similar solutions)
    - Finding the most reliable optimizer for a particular model
    - Debugging convergence issues

    Parameters
    ----------
    formula : str
        Model formula in lme4 syntax.
    data : pd.DataFrame
        Data containing the variables in the formula.
    optimizers : list[str], optional
        List of optimizer names to try. If None, uses a default set of
        robust optimizers: ["bobyqa", "Nelder-Mead", "L-BFGS-B"].
    REML : bool, default True
        Use REML estimation.
    verbose : int, default 0
        Verbosity level (0 = silent, 1 = show progress, 2 = show all output).
    **kwargs
        Additional arguments passed to lmer().

    Returns
    -------
    AllFitResult
        Object containing all fitted models and a comparison summary.

    Examples
    --------
    >>> import mixedlm as mlm
    >>> data = mlm.load_sleepstudy()
    >>> result = mlm.allFit("Reaction ~ Days + (Days|Subject)", data)
    >>> print(result)
    >>> result.summary

    >>> # Try specific optimizers
    >>> result = mlm.allFit(
    ...     "Reaction ~ Days + (1|Subject)",
    ...     data,
    ...     optimizers=["bobyqa", "L-BFGS-B", "Nelder-Mead", "Powell"]
    ... )

    Notes
    -----
    Similar to lme4's allFit() function in R. If all optimizers converge to
    different solutions, this may indicate optimization difficulties or
    model specification issues.

    See Also
    --------
    lmer : Fit linear mixed-effects model
    lmerControl : Control parameters for optimization
    """
    import pandas as pd

    from mixedlm.models.control import lmerControl
    from mixedlm.models.lmer import lmer

    if optimizers is None:
        optimizers = ["bobyqa", "Nelder-Mead", "L-BFGS-B"]

    results: dict[str, LmerResult | Exception] = {}

    if verbose >= 1:
        print(f"Fitting model with {len(optimizers)} optimizers...")

    for opt_name in optimizers:
        if verbose >= 1:
            print(f"  Trying {opt_name}...", end=" ")

        try:
            ctrl = lmerControl(optimizer=opt_name)
            result = lmer(formula, data, REML=REML, control=ctrl, verbose=0, **kwargs)
            results[opt_name] = result

            if verbose >= 1:
                status = "OK" if result.converged else "FAILED"
                print(f"{status} (deviance={result.deviance:.4f}, iter={result.n_iter})")

        except Exception as e:
            results[opt_name] = e
            if verbose >= 1:
                print(f"ERROR: {e}")

    summary_data = []
    for opt_name in results:
        result = results[opt_name]  # type: ignore[assignment]
        if isinstance(result, Exception):
            summary_data.append(
                {
                    "optimizer": opt_name,
                    "converged": False,
                    "deviance": np.nan,
                    "iterations": np.nan,
                    "gradient_norm": np.nan,
                    "at_boundary": np.nan,
                    "error": str(result),
                }
            )
        else:
            summary_data.append(
                {
                    "optimizer": opt_name,
                    "converged": result.converged,
                    "deviance": result.deviance,
                    "iterations": result.n_iter,
                    "gradient_norm": result.gradient_norm or np.nan,
                    "at_boundary": result.at_boundary,
                    "error": None,
                }
            )

    summary = pd.DataFrame(summary_data)

    successful_results = {
        name: res for name, res in results.items() if not isinstance(res, Exception)
    }

    if successful_results:
        best_optimizer = min(
            successful_results.keys(), key=lambda k: successful_results[k].deviance
        )
    else:
        best_optimizer = optimizers[0]

    return AllFitResult(results=results, summary=summary, best_optimizer=best_optimizer)
