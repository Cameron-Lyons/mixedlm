from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mixedlm.estimation.laplace import GLMMOptimizer
from mixedlm.families.base import Family
from mixedlm.families.binomial import Binomial
from mixedlm.formula.parser import parse_formula
from mixedlm.formula.terms import Formula
from mixedlm.matrices.design import build_model_matrices
from mixedlm.models.glmer import GlmerResult
from mixedlm.models.shared_utils import resolve_optional_vector

if TYPE_CHECKING:
    import pandas as pd

    from mixedlm.models.control import GlmerControl


class GlmerMod:
    def __init__(
        self,
        formula: str | Formula,
        data: pd.DataFrame,
        family: Family | None = None,
        verbose: int = 0,
        weights: NDArray[np.floating] | None = None,
        offset: NDArray[np.floating] | None = None,
        na_action: str | None = "omit",
        contrasts: dict[str, str | NDArray[np.floating]] | None = None,
        control: GlmerControl | None = None,
    ) -> None:
        from mixedlm.models.control import GlmerControl

        if isinstance(formula, str):
            self.formula = parse_formula(formula)
        else:
            self.formula = formula

        self.data = data
        self.family = family if family is not None else Binomial()
        self.verbose = verbose
        self.na_action = na_action
        self.contrasts = contrasts
        self.control = control if control is not None else GlmerControl()

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
        nAGQ: int = 1,
    ) -> GlmerResult:
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

        optimizer = GLMMOptimizer(
            self.matrices,
            self.family,
            verbose=self.verbose,
            nAGQ=nAGQ,
        )

        opt_result = optimizer.optimize(
            start=start,
            method=opt_method,
            maxiter=opt_maxiter,
            options=ctrl.optCtrl,
        )

        result = GlmerResult(
            formula=self.formula,
            matrices=self.matrices,
            family=self.family,
            theta=opt_result.theta,
            beta=opt_result.beta,
            u=opt_result.u,
            deviance=opt_result.deviance,
            converged=opt_result.converged,
            n_iter=opt_result.n_iter,
            nAGQ=nAGQ,
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


def glmer_nb(
    formula: str,
    data: pd.DataFrame,
    verbose: int = 0,
    nAGQ: int = 1,
    weights: NDArray[np.floating] | str | None = None,
    offset: NDArray[np.floating] | str | None = None,
    na_action: str | None = "omit",
    contrasts: dict[str, str | NDArray[np.floating]] | None = None,
    control: GlmerControl | None = None,
    theta: float = 1.0,
    **kwargs,
) -> GlmerResult:
    """Fit a negative binomial generalized linear mixed-effects model.

    This is a convenience wrapper around glmer() that uses the negative
    binomial family. It's equivalent to calling glmer() with
    family=NegativeBinomial(theta).

    Parameters
    ----------
    formula : str
        Model formula in lme4 syntax (e.g., "y ~ x + (1|group)").
    data : DataFrame
        Data containing the variables in the formula.
    verbose : int, default 0
        Verbosity level for optimization output.
    nAGQ : int, default 1
        Number of adaptive Gauss-Hermite quadrature points.
    weights : array-like, optional
        Prior weights for observations.
    offset : array-like, optional
        Offset term for the linear predictor.
    na_action : str, optional
        How to handle missing values ("omit", "exclude", "fail").
    contrasts : dict, optional
        Contrast specifications for categorical variables.
    control : GlmerControl, optional
        Control parameters for the optimizer.
    theta : float, default 1.0
        The theta (dispersion) parameter for the negative binomial
        distribution. Larger values indicate less overdispersion.
    **kwargs
        Additional arguments passed to the optimizer.

    Returns
    -------
    GlmerResult
        Fitted model result.

    Examples
    --------
    >>> result = glmer_nb("count ~ treatment + (1|subject)", data)

    >>> result = glmer_nb("count ~ x + (1|group)", data, theta=2.0)

    Notes
    -----
    The negative binomial distribution is useful for count data that
    exhibits overdispersion (variance > mean). The theta parameter
    controls the degree of overdispersion: as theta -> infinity, the
    negative binomial approaches the Poisson distribution.

    See Also
    --------
    glmer : General GLMM fitting function.
    NegativeBinomial : The negative binomial family class.
    """
    from mixedlm.families.negative_binomial import NegativeBinomial

    family = NegativeBinomial(theta=theta)
    return glmer(
        formula=formula,
        data=data,
        family=family,
        verbose=verbose,
        nAGQ=nAGQ,
        weights=weights,
        offset=offset,
        na_action=na_action,
        contrasts=contrasts,
        control=control,
        **kwargs,
    )


def glmer(
    formula: str,
    data: pd.DataFrame,
    family: Family | None = None,
    verbose: int = 0,
    nAGQ: int = 1,
    weights: NDArray[np.floating] | str | None = None,
    offset: NDArray[np.floating] | str | None = None,
    na_action: str | None = "omit",
    contrasts: dict[str, str | NDArray[np.floating]] | None = None,
    control: GlmerControl | None = None,
    **kwargs,
) -> GlmerResult:
    """Fit a generalized linear mixed-effects model.

    Parameters
    ----------
    formula : str
        Model formula in lme4 syntax (e.g., "y ~ x + (1|group)").
    data : DataFrame
        Data containing the variables in the formula.
    family : Family, optional
        GLM family (default: Binomial).
    verbose : int, default 0
        Verbosity level for optimization output.
    nAGQ : int, default 1
        Number of adaptive Gauss-Hermite quadrature points.
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
        Dictionary mapping variable names to contrast specifications.
        Values can be:
        - "treatment" (default): Treatment/dummy contrasts
        - "sum": Sum/deviation contrasts
        - "helmert": Helmert contrasts
        - "poly": Polynomial contrasts for ordered factors
        - A custom contrast matrix (NDArray of shape (n_levels, n_levels-1))
    control : GlmerControl, optional
        Control parameters for the optimizer. Use glmerControl() to create.
        If not provided, default settings are used.
    **kwargs
        Additional arguments passed to the optimizer (start, method, maxiter).

    Returns
    -------
    GlmerResult
        Fitted model result.

    Examples
    --------
    >>> result = glmer("y ~ x + (1|group)", data, family=Binomial())

    >>> from mixedlm import glmerControl
    >>> ctrl = glmerControl(optimizer="Nelder-Mead", maxiter=2000)
    >>> result = glmer("y ~ x + (1|group)", data, family=Binomial(), control=ctrl)
    """
    weights_arr = resolve_optional_vector(data, weights, "weights")
    offset_arr = resolve_optional_vector(data, offset, "offset")

    model = GlmerMod(
        formula,
        data,
        family=family,
        verbose=verbose,
        weights=weights_arr,
        offset=offset_arr,
        na_action=na_action,
        contrasts=contrasts,
        control=control,
    )
    return model.fit(nAGQ=nAGQ, **kwargs)
