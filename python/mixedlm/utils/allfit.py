from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any

from mixedlm.inference.allfit import AllFitResult, _run_allfit

if TYPE_CHECKING:
    import pandas as pd

    from mixedlm.models.control import LmerControl
    from mixedlm.models.lmer import LmerResult


def _fit_formula(
    formula: str,
    data: pd.DataFrame,
    REML: bool,
    control: LmerControl | None,
    kwargs: dict[str, Any],
    optimizer: str,
) -> LmerResult:
    from mixedlm.models.control import LmerControl
    from mixedlm.models.lmer import lmer

    control = replace(control if control is not None else LmerControl(), optimizer=optimizer)
    return lmer(formula, data, REML=REML, control=control, verbose=0, **kwargs)


def allFit(
    formula: str,
    data: pd.DataFrame,
    optimizers: list[str] | None = None,
    REML: bool = True,
    verbose: int = 0,
    n_jobs: int = 1,
    control: LmerControl | None = None,
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
        robust optimizers: ["COBYQA", "Nelder-Mead", "L-BFGS-B"].
    REML : bool, default True
        Use REML estimation.
    verbose : int, default 0
        Verbosity level (0 = silent, 1 or more = report each optimizer's outcome).
    n_jobs : int, default 1
        Number of worker processes, or -1 for all available CPUs. Workers are
        started without forking, so scripts need an ``if __name__ == "__main__":``
        guard.
    control : LmerControl, optional
        Control settings shared by every fit; only the optimizer is replaced.
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
    ...     optimizers=["COBYQA", "L-BFGS-B", "Nelder-Mead", "Powell"]
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
    if optimizers is None:
        optimizers = ["COBYQA", "Nelder-Mead", "L-BFGS-B"]
    if verbose >= 1:
        print(f"Fitting model with {len(optimizers)} optimizers...")
    fit = partial(_fit_formula, formula, data, REML, control, kwargs)
    return _run_allfit(fit, optimizers, n_jobs, verbose >= 1)
