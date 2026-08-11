from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult

from mixedlm.inference.allfit import AllFitResult


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
        robust optimizers: ["COBYQA", "Nelder-Mead", "L-BFGS-B"].
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
    from mixedlm.models.control import lmerControl
    from mixedlm.models.lmer import lmer

    if optimizers is None:
        optimizers = ["COBYQA", "Nelder-Mead", "L-BFGS-B"]

    fits: dict[str, LmerResult | GlmerResult | None] = {}
    errors: dict[str, str] = {}
    warnings: dict[str, list[str]] = {}

    if verbose >= 1:
        print(f"Fitting model with {len(optimizers)} optimizers...")

    for opt_name in optimizers:
        if verbose >= 1:
            print(f"  Trying {opt_name}...", end=" ")

        try:
            ctrl = lmerControl(optimizer=opt_name)
            fit_result = lmer(formula, data, REML=REML, control=ctrl, verbose=0, **kwargs)
            fits[opt_name] = fit_result
            warnings[opt_name] = []
            if not fit_result.converged:
                warnings[opt_name].append("Did not converge")
            if fit_result.isSingular():
                warnings[opt_name].append("Singular fit")

            if verbose >= 1:
                status = "OK" if fit_result.converged else "FAILED"
                print(f"{status} (deviance={fit_result.deviance:.4f}, iter={fit_result.n_iter})")

        except Exception as e:
            fits[opt_name] = None
            errors[opt_name] = str(e)
            warnings[opt_name] = []
            if verbose >= 1:
                print(f"ERROR: {e}")

    return AllFitResult(fits=fits, errors=errors, warnings=warnings)
