from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mixedlm.estimation.laplace import GLMMOptimizer
from mixedlm.families.base import Family
from mixedlm.families.binomial import Binomial
from mixedlm.formula.parser import parse_formula
from mixedlm.formula.terms import Formula
from mixedlm.matrices.design import build_model_matrices
from mixedlm.models.glmer import GlmerResult
from mixedlm.models.shared_utils import resolve_em_start, resolve_optional_vector
from mixedlm.utils.dataframe import ensure_dataframe

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
        transform_state: dict[str, Any] | None = None,
        factor_coding: str = "R",
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
            transform_state=transform_state,
            grouped_binomial=isinstance(self.family, Binomial),
            factor_coding=factor_coding,
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

        start = resolve_em_start(start, self.matrices, ctrl, self.verbose)

        optimizer = GLMMOptimizer(
            self.matrices,
            self.family,
            verbose=self.verbose,
            nAGQ=nAGQ,
            pirls_maxiter=ctrl.pirls_maxiter,
            pirls_tol=ctrl.tolPwrss,
            nAGQ0initStep=ctrl.nAGQ0initStep,
        )

        opt_options = ctrl.get_scipy_options(optimizer=opt_method, maxiter=opt_maxiter)
        opt_result = optimizer.optimize(
            start=start,
            method=opt_method,
            maxiter=opt_maxiter,
            options=opt_options,
            restart_edge=ctrl.restart_edge,
        )
        from dataclasses import replace

        from mixedlm.models.control import GlmerControl

        kept_method = opt_method
        if (
            method is None
            and opt_method == GlmerControl().optimizer
            and opt_result.converged
            and _at_boundary(opt_result.theta, self.matrices, ctrl.boundary_tol)
        ):
            # A derivative-free search can stop at a worse boundary optimum of a
            # singular fit. Check it once from the same start with a gradient
            # method and keep whichever fit reaches the lower deviance.
            # The check takes no optimizer-specific options meant for COBYQA.
            check_control = replace(ctrl, optCtrl={})
            second = optimizer.optimize(
                start=start,
                method="L-BFGS-B",
                maxiter=opt_maxiter,
                options=check_control.get_scipy_options(optimizer="L-BFGS-B", maxiter=opt_maxiter),
                restart_edge=ctrl.restart_edge,
            )
            if second.converged and second.deviance < opt_result.deviance - 1e-6:
                # The result records the method whose estimates were kept.
                opt_result, kept_method = second, "L-BFGS-B"

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
            pirls_converged=opt_result.pirls_converged,
            pirls_maxiter=optimizer.pirls_maxiter,
            pirls_tol=optimizer.pirls_tol,
            joint_fit=opt_result.joint_fit,
            message=opt_result.message,
            optimizer=kept_method,
            control=deepcopy(ctrl),
        )

        if ctrl.check_conv and not result.converged:
            warnings.warn(
                (
                    "Model failed to converge: the inner PIRLS solver did not converge. "
                    "Consider increasing pirls_maxiter or inspecting "
                    "the response and model specification."
                    if not result.pirls_converged
                    else "Model failed to converge. Consider increasing maxiter or "
                    "trying a different optimizer."
                ),
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
    formula: Formula | str,
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
    formula : Formula or str
        Parsed formula or model formula in lme4 syntax (e.g., "y ~ x + (1|group)").
    data : DataFrame
        Data containing the variables in the formula.
    verbose : int, default 0
        Verbosity level for optimization output.
    nAGQ : int, default 1
        Fitting approximation: 0 uses joint PIRLS with theta-only optimization;
        1 jointly optimizes theta and beta using Laplace approximation; larger
        integers use adaptive quadrature for a single scalar random-effect term.
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


def _at_boundary(theta: NDArray[np.floating], matrices: Any, tolerance: float) -> bool:
    """Whether a covariance factor's diagonal sits at its zero bound."""
    from mixedlm.models.shared_utils import is_singular_theta

    return bool(is_singular_theta(theta, matrices.random_structures, tolerance))


def glmer(
    formula: Formula | str,
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
    formula : Formula or str
        Parsed formula or model formula in lme4 syntax (e.g., "y ~ x + (1|group)").
    data : DataFrame
        Data containing the variables in the formula.
    family : Family, optional
        GLM family (default: Binomial).
    verbose : int, default 0
        Verbosity level for optimization output.
    nAGQ : int, default 1
        Fitting approximation: 0 uses joint PIRLS with theta-only optimization;
        1 jointly optimizes theta and beta using Laplace approximation; larger
        integers use adaptive quadrature for a single scalar random-effect term.
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
    parsed_formula = parse_formula(formula) if isinstance(formula, str) else formula
    required_columns = set(parsed_formula.all_variables)
    if isinstance(weights, str):
        required_columns.add(weights)
    if isinstance(offset, str):
        required_columns.add(offset)
    data = ensure_dataframe(data, columns=sorted(required_columns))

    weights_arr = resolve_optional_vector(data, weights, "weights")
    offset_arr = resolve_optional_vector(data, offset, "offset")

    model = GlmerMod(
        parsed_formula,
        data,
        family=family,
        verbose=verbose,
        weights=weights_arr,
        offset=offset_arr,
        na_action=na_action,
        contrasts=contrasts,
        control=control,
        # Formula refits on the fitted rows reuse the fitted transform parameters.
        transform_state=kwargs.pop("_transform_state", None),
        # Refits of models saved by mixedlm 1.3 and earlier keep their coding.
        factor_coding=kwargs.pop("_factor_coding", "R"),
    )
    return model.fit(nAGQ=nAGQ, **kwargs)
