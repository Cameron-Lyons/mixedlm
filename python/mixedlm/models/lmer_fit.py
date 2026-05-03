from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mixedlm.estimation.reml import LMMOptimizer
from mixedlm.formula.parser import parse_formula
from mixedlm.formula.terms import Formula
from mixedlm.matrices.design import build_model_matrices
from mixedlm.models.lmer import LmerResult
from mixedlm.models.shared_utils import resolve_optional_vector

if TYPE_CHECKING:
    import pandas as pd

    from mixedlm.models.control import LmerControl


class LmerMod:
    def __init__(
        self,
        formula: str | Formula,
        data: pd.DataFrame,
        REML: bool = True,
        verbose: int = 0,
        weights: NDArray[np.floating] | None = None,
        offset: NDArray[np.floating] | None = None,
        na_action: str | None = "omit",
        contrasts: dict[str, str | NDArray[np.floating]] | None = None,
        control: LmerControl | None = None,
    ) -> None:
        from mixedlm.models.control import LmerControl

        if isinstance(formula, str):
            self.formula = parse_formula(formula)
        else:
            self.formula = formula

        self.data = data
        self.REML = REML
        self.verbose = verbose
        self.na_action = na_action
        self.contrasts = contrasts
        self.control = control if control is not None else LmerControl()

        self.matrices = build_model_matrices(
            self.formula,
            self.data,
            weights=weights,
            offset=offset,
            na_action=na_action,
            contrasts=contrasts,
        )

    def fit(
        self,
        start: NDArray[np.floating] | None = None,
        method: str | None = None,
        maxiter: int | None = None,
    ) -> LmerResult:
        import warnings

        from mixedlm.models.checks import run_model_checks

        ctrl = self.control
        opt_method = method if method is not None else ctrl.optimizer
        opt_maxiter = maxiter if maxiter is not None else ctrl.maxiter

        self.matrices, self._dropped_cols = run_model_checks(self.matrices, ctrl)

        if ctrl.em_init and start is None:
            from mixedlm.estimation.em_reml import em_reml_simple

            try:
                em_result = em_reml_simple(
                    self.matrices,
                    max_iter=ctrl.em_maxiter,
                    verbose=max(self.verbose - 1, 0),
                )
                start = em_result.theta
            except (NotImplementedError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                warnings.warn(
                    f"EM initialization failed ({type(exc).__name__}: {exc}). "
                    "Falling back to optimizer defaults.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        optimizer = LMMOptimizer(
            self.matrices,
            REML=self.REML,
            verbose=self.verbose,
            use_rust=ctrl.use_rust,
        )

        opt_result = optimizer.optimize(
            start=start,
            method=opt_method,
            maxiter=opt_maxiter,
            options=ctrl.optCtrl,
        )

        result = LmerResult(
            formula=self.formula,
            matrices=self.matrices,
            theta=opt_result.theta,
            beta=opt_result.beta,
            sigma=opt_result.sigma,
            u=opt_result.u,
            deviance=opt_result.deviance,
            REML=self.REML,
            converged=opt_result.converged,
            n_iter=opt_result.n_iter,
            gradient_norm=opt_result.gradient_norm,
            at_boundary=opt_result.at_boundary,
            message=opt_result.message,
            function_evals=opt_result.function_evals,
        )

        if ctrl.check_conv and not result.converged:
            warnings.warn(
                "Model failed to converge. Consider increasing maxiter or "
                "trying a different optimizer.",
                category=UserWarning,
                stacklevel=2,
            )

        if ctrl.check_singular and result.isSingular(tol=ctrl.boundary_tol):
            warnings.warn(
                "Model is singular (boundary fit). Some variance components "
                "are estimated as zero or near-zero.",
                category=UserWarning,
                stacklevel=2,
            )

        return result


def lmer(
    formula: str,
    data: pd.DataFrame,
    REML: bool = True,
    verbose: int = 0,
    weights: NDArray[np.floating] | str | None = None,
    offset: NDArray[np.floating] | str | None = None,
    na_action: str | None = "omit",
    contrasts: dict[str, str | NDArray[np.floating]] | None = None,
    control: LmerControl | None = None,
    **kwargs,
) -> LmerResult:
    """Fit a linear mixed-effects model.

    Parameters
    ----------
    formula : str
        Model formula in lme4 syntax (e.g., "y ~ x + (1|group)").
    data : DataFrame
        Data containing the variables in the formula.
    REML : bool, default True
        Use REML estimation. Set to False for ML estimation.
    verbose : int, default 0
        Verbosity level for optimization output.
    weights : array-like, optional
        Prior weights for observations.
    offset : array-like, optional
        Offset term for the linear predictor.
    na_action : str, optional
        How to handle missing values. Options:
        - "omit" (default): Remove rows with any NA values
        - "exclude": Like omit, but fitted/residuals return NA for removed rows
        - "fail": Raise an error if any NA values are present
    contrasts : dict, optional
        Contrast coding for categorical variables. Keys are variable names,
        values can be:
        - "treatment" (default): Treatment/dummy contrasts
        - "sum": Sum/deviation contrasts
        - "helmert": Helmert contrasts
        - "poly": Polynomial contrasts for ordered factors
        - A custom contrast matrix (NDArray)
    control : LmerControl, optional
        Control parameters for the optimizer. Use lmerControl() to create.
        If not provided, default settings are used.
    **kwargs
        Additional arguments passed to the optimizer (start, method, maxiter).

    Returns
    -------
    LmerResult
        Fitted model result.

    Examples
    --------
    >>> result = lmer("y ~ x + (1|group)", data)

    >>> from mixedlm import lmerControl
    >>> ctrl = lmerControl(optimizer="Nelder-Mead", maxiter=2000)
    >>> result = lmer("y ~ x + (1|group)", data, control=ctrl)
    """
    weights_arr = resolve_optional_vector(data, weights, "weights")
    offset_arr = resolve_optional_vector(data, offset, "offset")

    model = LmerMod(
        formula,
        data,
        REML=REML,
        verbose=verbose,
        weights=weights_arr,
        offset=offset_arr,
        na_action=na_action,
        contrasts=contrasts,
        control=control,
    )
    return model.fit(**kwargs)
