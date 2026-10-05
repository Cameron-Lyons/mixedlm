from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mixedlm.estimation.joint_glmm import JointGLMMObjective
    from mixedlm.estimation.laplace import GLMMOptimizer
    from mixedlm.families.base import Family
    from mixedlm.formula.terms import Formula
    from mixedlm.models.control import GlmerControl, LmerControl
    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult

from mixedlm.estimation.reml import (
    AUTO_OPTIMIZER,
    LMMOptimizer,
    _build_lambda,
    _build_theta_bounds,
    _count_theta,
)
from mixedlm.formula.parser import parse_formula
from mixedlm.formula.terms import InteractionTerm, PowerTerm, VariableTerm
from mixedlm.matrices.design import ModelMatrices, build_model_matrices, build_random_matrix


def _coerce_formula(formula: Formula | str) -> Formula:
    """Return an existing Formula or parse its string representation."""
    return parse_formula(formula) if isinstance(formula, str) else formula


@dataclass
class _ParsedFormulaBase:
    """Shared parsed-formula container for mixed model modular APIs."""

    formula: Formula
    matrices: ModelMatrices

    @property
    def X(self) -> NDArray[np.floating]:
        """Fixed effects design matrix."""
        return self.matrices.X

    @property
    def Z(self):
        """Random effects design matrix (sparse)."""
        return self.matrices.Z

    @property
    def y(self) -> NDArray[np.floating]:
        """Response vector."""
        return self.matrices.y

    @property
    def n_obs(self) -> int:
        """Number of observations."""
        return self.matrices.n_obs

    @property
    def n_fixed(self) -> int:
        """Number of fixed effects."""
        return self.matrices.n_fixed

    @property
    def n_random(self) -> int:
        """Number of random effects."""
        return self.matrices.n_random

    @property
    def n_theta(self) -> int:
        """Number of variance component parameters."""
        return _count_theta(self.matrices.random_structures)


@dataclass
class LmerParsedFormula(_ParsedFormulaBase):
    """Result of lFormula - parsed formula and model matrices for LMM.

    This class contains all the information needed to construct the
    deviance function and fit a linear mixed model.

    Attributes
    ----------
    formula : Formula
        The parsed formula object.
    matrices : ModelMatrices
        Model matrices including X (fixed effects), Z (random effects),
        y (response), weights, offset, and random effect structures.
    REML : bool
        Whether to use REML estimation.
    """

    REML: bool


@dataclass
class GlmerParsedFormula(_ParsedFormulaBase):
    """Result of glFormula - parsed formula and model matrices for GLMM.

    This class contains all the information needed to construct the
    deviance function and fit a generalized linear mixed model.

    Attributes
    ----------
    formula : Formula
        The parsed formula object.
    matrices : ModelMatrices
        Model matrices including X (fixed effects), Z (random effects),
        y (response), weights, offset, and random effect structures.
    family : Family
        The GLM family (e.g., Binomial, Poisson).
    """

    family: Family


@dataclass
class LmerDevfun:
    """Deviance function for linear mixed models.

    This class wraps the profiled deviance function and provides
    methods for evaluation and optimization.

    Attributes
    ----------
    parsed : LmerParsedFormula
        The parsed formula result from lFormula.
    optimizer : LMMOptimizer
        The optimizer object for computing deviance.
    """

    parsed: LmerParsedFormula
    optimizer: LMMOptimizer
    control: LmerControl | None = None

    def __call__(self, theta: NDArray[np.floating]) -> float:
        """Evaluate the deviance function at theta.

        Parameters
        ----------
        theta : NDArray
            Variance component parameters (relative covariance factors).

        Returns
        -------
        float
            The profiled deviance (or REML criterion).
        """
        return self.optimizer.objective(theta)

    def get_start(self) -> NDArray[np.floating]:
        """Get default starting values for theta.

        Returns
        -------
        NDArray
            Starting values (ones by default).
        """
        return self.optimizer.get_start_theta()

    def get_bounds(self) -> list[tuple[float | None, float | None]]:
        """Get bounds for theta parameters.

        Diagonal elements of the Cholesky factor must be non-negative.

        Returns
        -------
        list of tuple
            Bounds for each theta parameter.
        """
        return _build_theta_bounds(self.parsed.matrices.random_structures, self.parsed.n_theta)


@dataclass
class GlmerDevfun:
    """Deviance function for generalized linear mixed models.

    This class wraps the Laplace or adaptive-quadrature deviance function
    and provides methods for evaluation and optimization. Full [theta, beta]
    vectors evaluate the joint likelihood; theta-only vectors estimate beta
    through PIRLS at the configured quadrature order.

    Repeated joint calls reuse preparation until the optimizer or its solver
    settings change. Treat the optimizer's model arrays and family as immutable.

    Attributes
    ----------
    parsed : GlmerParsedFormula
        The parsed formula result from glFormula.
    optimizer : GLMMOptimizer
        The optimizer object for computing deviance.
    """

    parsed: GlmerParsedFormula
    optimizer: GLMMOptimizer
    control: GlmerControl | None = None
    _joint_cache: tuple[GLMMOptimizer, JointGLMMObjective] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def _joint_objective(self) -> JointGLMMObjective:
        optimizer = self.optimizer
        cached = self._joint_cache
        if cached is not None and cached[0] is optimizer:
            from mixedlm.estimation.laplace import _validate_quadrature
            from mixedlm.estimation.pirls_control import validate_pirls_controls

            # Settings remain editable. Validate even when equality would hide
            # an invalid replacement such as nAGQ=True after nAGQ=1. A cache miss
            # is validated by the objective's constructor instead.
            _validate_quadrature(optimizer.nAGQ, optimizer.matrices)
            validate_pirls_controls(optimizer.pirls_maxiter, optimizer.pirls_tol)
            objective = cached[1]
            if (objective.nAGQ, objective.pirls_maxiter, objective.pirls_tol) == (
                optimizer.nAGQ,
                optimizer.pirls_maxiter,
                optimizer.pirls_tol,
            ):
                return objective
        objective = optimizer.joint_objective()
        self._joint_cache = (optimizer, objective)
        return objective

    def __call__(self, theta: NDArray[np.floating]) -> float:
        """Evaluate theta with PIRLS beta, or a full [theta, beta] vector.

        Parameters
        ----------
        theta : NDArray
            Covariance parameters, optionally followed by fixed coefficients.

        Returns
        -------
        float
            The deviance approximation at the configured quadrature setting.
        """
        if len(theta) == self.parsed.n_theta + self.parsed.n_fixed and self.parsed.n_fixed:
            return self._joint_objective()(theta)
        return self.optimizer.objective(theta)

    def get_start(self, *, joint: bool = False) -> NDArray[np.floating]:
        """Get starting theta, or [theta, beta] when joint=True.

        Returns
        -------
        NDArray
            Starting covariance parameters, optionally followed by PIRLS beta.
        """
        theta = self.optimizer.get_start_theta()
        if joint:
            _, beta, _, _ = self.optimizer._final_evaluation_with_status(theta, nAGQ=1)
            return np.r_[theta, beta]
        return theta

    def get_bounds(self, *, joint: bool = False) -> list[tuple[float | None, float | None]]:
        """Get theta bounds, optionally followed by unbounded beta entries.

        Returns
        -------
        list of tuple
            Bounds for each theta parameter.
        """
        bounds = _build_theta_bounds(self.parsed.matrices.random_structures, self.parsed.n_theta)
        return bounds + [(None, None)] * self.parsed.n_fixed if joint else bounds


@dataclass
class OptimizeResult:
    """Result of optimization for mixed models.

    Attributes
    ----------
    theta : NDArray
        Optimized variance component parameters.
    deviance : float
        Final deviance value.
    converged : bool
        Whether optimization converged.
    n_iter : int
        Number of iterations.
    message : str
        Optimization message.
    nAGQ : int, optional
        Quadrature setting recorded by optimizeGlmer; omitted for linear models
        and custom optimization results unless supplied explicitly.
    beta : NDArray, optional
        Jointly optimized fixed coefficients for GLMMs. None retains theta-only
        PIRLS extraction in mkGlmerMod and is the default for custom results.
    optimizer : str
        Optimization method, recorded on the fitted model for checkConv.
    """

    theta: NDArray[np.floating]
    deviance: float
    converged: bool
    n_iter: int
    message: str
    nAGQ: int | None = None
    pirls_converged: bool | None = None
    beta: NDArray[np.floating] | None = None
    optimizer: str = ""


def lFormula(
    formula: Formula | str,
    data: pd.DataFrame,
    REML: bool = True,
    weights: NDArray[np.floating] | None = None,
    offset: NDArray[np.floating] | None = None,
    na_action: str | None = "omit",
    contrasts: dict[str, str | NDArray[np.floating]] | None = None,
) -> LmerParsedFormula:
    """Parse a formula and create model matrices for a linear mixed model.

    This is the first step in the modular interface for fitting LMMs.
    It parses the formula and builds the design matrices without
    performing any optimization.

    Parameters
    ----------
    formula : Formula or str
        Parsed formula or model formula in lme4 syntax (e.g., "y ~ x + (1|group)").
    data : DataFrame
        Data containing the variables in the formula.
    REML : bool, default True
        Whether to use REML estimation (stored for later use).
    weights : array-like, optional
        Prior weights for observations.
    offset : array-like, optional
        Offset term for the linear predictor.
    na_action : str, optional
        How to handle missing values: "omit", "exclude", or "fail".
    contrasts : dict, optional
        Contrast coding for categorical variables.

    Returns
    -------
    LmerParsedFormula
        Object containing the parsed formula and model matrices.

    Examples
    --------
    >>> parsed = lFormula("y ~ x + (1|group)", data)
    >>> parsed.X.shape  # Fixed effects design matrix
    >>> parsed.n_theta  # Number of variance parameters

    See Also
    --------
    mkLmerDevfun : Create deviance function from parsed formula.
    optimizeLmer : Optimize the deviance function.
    mkLmerMod : Create final model from optimization results.
    """
    parsed_formula = _coerce_formula(formula)
    matrices = build_model_matrices(
        parsed_formula,
        data,
        weights=weights,
        offset=offset,
        na_action=na_action,
        contrasts=contrasts,
    )

    return LmerParsedFormula(
        formula=parsed_formula,
        matrices=matrices,
        REML=REML,
    )


def glFormula(
    formula: Formula | str,
    data: pd.DataFrame,
    family: Family | None = None,
    weights: NDArray[np.floating] | None = None,
    offset: NDArray[np.floating] | None = None,
    na_action: str | None = "omit",
    contrasts: dict[str, str | NDArray[np.floating]] | None = None,
) -> GlmerParsedFormula:
    """Parse a formula and create model matrices for a generalized linear mixed model.

    This is the first step in the modular interface for fitting GLMMs.
    It parses the formula and builds the design matrices without
    performing any optimization.

    Parameters
    ----------
    formula : Formula or str
        Parsed formula or model formula in lme4 syntax (e.g., "y ~ x + (1|group)").
    data : DataFrame
        Data containing the variables in the formula.
    family : Family, optional
        GLM family (default: Binomial).
    weights : array-like, optional
        Prior weights for observations.
    offset : array-like, optional
        Offset term for the linear predictor.
    na_action : str, optional
        How to handle missing values: "omit", "exclude", or "fail".
    contrasts : dict, optional
        Contrast coding for categorical variables.

    Returns
    -------
    GlmerParsedFormula
        Object containing the parsed formula, model matrices, and family.

    Examples
    --------
    >>> from mixedlm.families import Binomial
    >>> parsed = glFormula("y ~ x + (1|group)", data, family=Binomial())
    >>> parsed.X.shape  # Fixed effects design matrix
    >>> parsed.family   # The GLM family

    See Also
    --------
    mkGlmerDevfun : Create deviance function from parsed formula.
    optimizeGlmer : Optimize the deviance function.
    """
    from mixedlm.families import Binomial

    parsed_formula = _coerce_formula(formula)
    if family is None:
        family = Binomial()

    matrices = build_model_matrices(
        parsed_formula,
        data,
        weights=weights,
        offset=offset,
        na_action=na_action,
        contrasts=contrasts,
        grouped_binomial=isinstance(family, Binomial),
    )

    return GlmerParsedFormula(
        formula=parsed_formula,
        matrices=matrices,
        family=family,
    )


def mkLmerDevfun(
    parsed: LmerParsedFormula,
    verbose: int = 0,
    control: LmerControl | None = None,
) -> LmerDevfun:
    """Create the deviance function for a linear mixed model.

    This is the second step in the modular interface. It creates
    the objective function that will be minimized to fit the model.

    Parameters
    ----------
    parsed : LmerParsedFormula
        Result from lFormula.
    verbose : int, default 0
        Verbosity level for optimization output.
    control : LmerControl, optional
        Control parameters for the optimizer.

    Returns
    -------
    LmerDevfun
        Callable deviance function object.

    Examples
    --------
    >>> parsed = lFormula("y ~ x + (1|group)", data)
    >>> devfun = mkLmerDevfun(parsed)
    >>> devfun.get_start()  # Get starting values
    >>> devfun(theta)  # Evaluate deviance at theta

    See Also
    --------
    lFormula : Parse formula and create model matrices.
    optimizeLmer : Optimize the deviance function.
    """
    from mixedlm.models.control import LmerControl

    if control is None:
        control = LmerControl()

    optimizer = LMMOptimizer(
        parsed.matrices,
        REML=parsed.REML,
        verbose=verbose,
        use_rust=control.use_rust,
    )

    return LmerDevfun(parsed=parsed, optimizer=optimizer, control=control)


def mkGlmerDevfun(
    parsed: GlmerParsedFormula,
    verbose: int = 0,
    control: GlmerControl | None = None,
    *,
    nAGQ: int = 1,
) -> GlmerDevfun:
    """Create the deviance function for a generalized linear mixed model.

    This is the second step in the modular interface for GLMMs. It creates
    the Laplace or adaptive-quadrature objective that will be minimized.

    Parameters
    ----------
    parsed : GlmerParsedFormula
        Result from glFormula.
    verbose : int, default 0
        Verbosity level for optimization output.
    control : GlmerControl, optional
        Control parameters for the optimizer.
    nAGQ : int, default 1
        Nonnegative fitting order: 0 for the theta-only PIRLS approximation,
        1 for joint Laplace fitting. Values above one use adaptive quadrature
        and require a single random-effect term with one coefficient per group.

    Returns
    -------
    GlmerDevfun
        Callable deviance function object.

    Examples
    --------
    >>> from mixedlm.families import Binomial
    >>> parsed = glFormula("y ~ x + (1|group)", data, family=Binomial())
    >>> devfun = mkGlmerDevfun(parsed)
    >>> devfun.get_start()  # Get starting values

    See Also
    --------
    glFormula : Parse formula and create model matrices.
    optimizeGlmer : Optimize the deviance function.
    """
    from mixedlm.estimation.laplace import GLMMOptimizer
    from mixedlm.models.control import GlmerControl

    if control is None:
        control = GlmerControl()

    optimizer = GLMMOptimizer(
        parsed.matrices,
        parsed.family,
        verbose=verbose,
        nAGQ=nAGQ,
        pirls_maxiter=control.pirls_maxiter,
        pirls_tol=control.tolPwrss,
        nAGQ0initStep=control.nAGQ0initStep,
    )

    return GlmerDevfun(parsed=parsed, optimizer=optimizer, control=control)


def optimizeLmer(
    devfun: LmerDevfun,
    start: NDArray[np.floating] | None = None,
    method: str = "L-BFGS-B",
    maxiter: int = 1000,
    verbose: int = 0,
    *,
    restart_edge: bool | None = None,
    use_analytic_gradient: bool | None = None,
) -> OptimizeResult:
    """Optimize the deviance function for a linear mixed model.

    This is the third step in the modular interface. It minimizes
    the deviance function to find optimal variance components.
    Tolerances and optCtrl from the deviance callable's control are applied
    to the requested method; optCtrl overrides generated solver options.

    Parameters
    ----------
    devfun : LmerDevfun
        Deviance function from mkLmerDevfun.
    start : NDArray, optional
        Starting values for theta. If None, uses default.
    method : str, default "L-BFGS-B"
        Optimization method (passed to scipy.optimize.minimize), or "auto" for
        lmer's exact-gradient L-BFGS-B with a COBYQA fallback.
    maxiter : int, default 1000
        Base iteration limit, unless overridden by the control's optCtrl.
        TNC and COBYLA use this as their function evaluation limit.
    verbose : int, default 0
        Verbosity level.
    restart_edge : bool or None, default None
        Check zero and near-zero covariance scales for likelihood improvement.
        None uses the control supplied to mkLmerDevfun (True by default).
        An explicit boolean overrides the control for this optimization call.
        Restarts use the requested optimizer and its remaining iteration budget.
    use_analytic_gradient : bool or None, default None
        Use native analytic gradients with supported optimizers. None uses the
        control supplied to mkLmerDevfun (False by default).

    Returns
    -------
    OptimizeResult
        Optimization result containing theta, deviance, and convergence info.

    Examples
    --------
    >>> parsed = lFormula("y ~ x + (1|group)", data)
    >>> devfun = mkLmerDevfun(parsed)
    >>> opt = optimizeLmer(devfun)
    >>> opt.theta  # Optimized variance parameters
    >>> opt.converged  # Did optimization converge?

    See Also
    --------
    mkLmerDevfun : Create deviance function.
    mkLmerMod : Create final model from optimization results.
    """
    from mixedlm.estimation.optimizers import run_optimizer

    if start is None:
        start = devfun.get_start()

    bounds = devfun.get_bounds()
    options = (
        devfun.control.get_scipy_options(optimizer=method, maxiter=maxiter)
        if devfun.control is not None
        else {"maxiter": maxiter}
    )
    if restart_edge is None:
        restart_edge = devfun.control.restart_edge if devfun.control is not None else True
    if use_analytic_gradient is None:
        use_analytic_gradient = (
            devfun.control.use_analytic_gradient if devfun.control is not None else False
        )
    objective, gradient = devfun.optimizer._optimization_functions(method, use_analytic_gradient)
    if gradient is None or type(devfun).__call__ is not LmerDevfun.__call__:
        # Custom callables may add terms absent from the native derivative.
        objective, gradient = devfun, None

    callback: Callable[[NDArray[np.floating]], None] | None = None
    if verbose > 0:

        def callback(x: NDArray[np.floating]) -> None:
            dev = objective(x)
            print(f"theta = {x}, deviance = {dev:.6f}")

    if method == AUTO_OPTIMIZER and type(devfun).__call__ is LmerDevfun.__call__:
        result, method = devfun.optimizer._optimize_auto(
            start, bounds, options, callback, restart_edge
        )
    else:
        if method == AUTO_OPTIMIZER:
            # Custom callables have no exact gradient for the L-BFGS-B stage.
            method = "COBYQA"
        result = run_optimizer(
            objective,
            start,
            method=method,
            bounds=bounds,
            options=options,
            callback=callback,
            jac=gradient,
            restart_edge=restart_edge,
        )

    return OptimizeResult(
        theta=result.x,
        deviance=result.fun,
        converged=result.success,
        n_iter=result.nit,
        message=result.message if hasattr(result, "message") else "",
        optimizer=method,
    )


def optimizeGlmer(
    devfun: GlmerDevfun,
    start: NDArray[np.floating] | None = None,
    method: str = "L-BFGS-B",
    maxiter: int = 1000,
    verbose: int = 0,
) -> OptimizeResult:
    """Optimize the deviance function for a generalized linear mixed model.

    This is the third step in the modular interface for GLMMs. At nAGQ>=1,
    optimize both theta and beta, optionally initializing with a theta-only fit.

    Parameters
    ----------
    devfun : GlmerDevfun
        Deviance function from mkGlmerDevfun.
    start : NDArray, optional
        Starting values for theta. If None, uses default.
    method : str, default "L-BFGS-B"
        Optimization method (passed to scipy.optimize.minimize).
    maxiter : int, default 1000
        Maximum iterations per outer optimization stage.
    verbose : int, default 0
        Verbosity level.

    Returns
    -------
    OptimizeResult
        Theta, joint beta (when applicable), deviance, convergence information,
        and the total iteration count across optimization stages.

    Examples
    --------
    >>> from mixedlm.families import Binomial
    >>> parsed = glFormula("y ~ x + (1|group)", data, family=Binomial())
    >>> devfun = mkGlmerDevfun(parsed)
    >>> opt = optimizeGlmer(devfun)
    >>> opt.theta  # Optimized variance parameters

    See Also
    --------
    mkGlmerDevfun : Create deviance function.
    """
    from copy import copy

    optimizer = copy(devfun.optimizer)
    optimizer.verbose = verbose
    options = (
        devfun.control.get_scipy_options(optimizer=method, maxiter=maxiter)
        if devfun.control is not None
        else None
    )
    result = optimizer.optimize(
        start=start,
        method=method,
        maxiter=maxiter,
        options=options,
        restart_edge=devfun.control.restart_edge if devfun.control is not None else True,
    )
    message = result.message
    if not result.pirls_converged:
        message += "; inner PIRLS solver did not converge"
    return OptimizeResult(
        theta=result.theta,
        beta=result.beta if result.joint_fit else None,
        deviance=result.deviance,
        converged=result.converged,
        pirls_converged=result.pirls_converged,
        n_iter=result.n_iter,
        message=message,
        nAGQ=optimizer.nAGQ,
        optimizer=method,
    )


def mkLmerMod(
    devfun: LmerDevfun,
    opt: OptimizeResult,
) -> LmerResult:
    """Create an LmerResult from optimization results.

    This is the final step in the modular interface. It constructs
    the fitted model object from the deviance function and optimization
    results.

    Parameters
    ----------
    devfun : LmerDevfun
        Deviance function from mkLmerDevfun.
    opt : OptimizeResult
        Optimization result from optimizeLmer.

    Returns
    -------
    LmerResult
        The fitted model result with all parameter estimates.

    Examples
    --------
    >>> parsed = lFormula("y ~ x + (1|group)", data)
    >>> devfun = mkLmerDevfun(parsed)
    >>> opt = optimizeLmer(devfun)
    >>> result = mkLmerMod(devfun, opt)
    >>> result.fixef()  # Fixed effects estimates
    >>> result.ranef()  # Random effects predictions

    See Also
    --------
    lFormula : Parse formula.
    mkLmerDevfun : Create deviance function.
    optimizeLmer : Optimize deviance.
    """
    from mixedlm.models.lmer import LmerResult

    evaluation = devfun.optimizer._final_evaluation(opt.theta)

    return LmerResult(
        formula=devfun.parsed.formula,
        matrices=devfun.parsed.matrices,
        theta=opt.theta,
        beta=evaluation.beta,
        sigma=evaluation.sigma,
        u=evaluation.u,
        deviance=evaluation.deviance,
        REML=devfun.parsed.REML,
        converged=opt.converged,
        n_iter=opt.n_iter,
        message=opt.message,
        optimizer=opt.optimizer,
    )


def mkGlmerMod(
    devfun: GlmerDevfun,
    opt: OptimizeResult,
    nAGQ: int | None = None,
) -> GlmerResult:
    """Create a GlmerResult from optimization results.

    This is the final step in the modular interface for GLMMs.

    Parameters
    ----------
    devfun : GlmerDevfun
        Deviance function from mkGlmerDevfun.
    opt : OptimizeResult
        Optimization result from optimizeGlmer.
    nAGQ : int, optional
        Must match the quadrature used for optimization. By default, inherit
        the optimization result's setting, or the deviance function's setting
        for a custom optimization result without quadrature metadata.

    Returns
    -------
    GlmerResult
        The fitted model result.

    Examples
    --------
    >>> from mixedlm.families import Binomial
    >>> parsed = glFormula("y ~ x + (1|group)", data, family=Binomial())
    >>> devfun = mkGlmerDevfun(parsed)
    >>> opt = optimizeGlmer(devfun)
    >>> result = mkGlmerMod(devfun, opt)
    >>> result.fixef()  # Fixed effects estimates

    See Also
    --------
    glFormula : Parse formula.
    mkGlmerDevfun : Create deviance function.
    optimizeGlmer : Optimize deviance.
    """
    from mixedlm.estimation.laplace import _validate_quadrature
    from mixedlm.models.glmer import GlmerResult

    fitted_nAGQ = opt.nAGQ if opt.nAGQ is not None else devfun.optimizer.nAGQ
    nAGQ = fitted_nAGQ if nAGQ is None else nAGQ
    _validate_quadrature(nAGQ, devfun.parsed.matrices)
    if nAGQ != fitted_nAGQ:
        raise ValueError(
            "nAGQ must match the setting used for optimization; "
            "create a deviance function with the requested nAGQ and optimize it again"
        )
    deviance, beta, u, pirls_converged = devfun.optimizer._final_evaluation_with_status(
        opt.theta, nAGQ=nAGQ, beta=opt.beta
    )

    return GlmerResult(
        formula=devfun.parsed.formula,
        matrices=devfun.parsed.matrices,
        family=devfun.parsed.family,
        theta=opt.theta,
        beta=beta,
        u=u,
        deviance=deviance,
        converged=bool(opt.converged and pirls_converged),
        pirls_converged=pirls_converged,
        pirls_maxiter=devfun.optimizer.pirls_maxiter,
        pirls_tol=devfun.optimizer.pirls_tol,
        joint_fit=opt.beta is not None,
        n_iter=opt.n_iter,
        nAGQ=nAGQ,
        message=opt.message,
        optimizer=opt.optimizer,
    )


@dataclass
class ReTrms:
    """Random effects terms structure.

    This class contains the components needed to construct the random
    effects portion of a mixed model. It mirrors the structure returned
    by lme4's mkReTrms function in R.

    Attributes
    ----------
    Zt : sparse matrix
        Transpose of the random effects design matrix (q x n).
    theta : ndarray
        Initial values for the variance component parameters.
    Lind : ndarray
        Index into theta for each element of the Lambda template.
    Gp : list
        Group pointers for each random effect term.
    flist : dict
        Dictionary of factor levels for each grouping factor.
    cnms : dict
        Dictionary of column names for each random effect term.
    nl : list
        Number of levels for each grouping factor.
    """

    Zt: object
    theta: NDArray[np.floating]
    Lind: NDArray[np.int_]
    Gp: list[int]
    flist: dict[str, NDArray]
    cnms: dict[str, list[str]]
    nl: list[int]
    # The source design lets mkNewReTrms encode new data exactly as mkReTrms did.
    _formula: Formula | None = field(default=None, repr=False, compare=False)
    _matrices: ModelMatrices | None = field(default=None, repr=False, compare=False)


def mkReTrms(
    formula: str,
    data: pd.DataFrame,
) -> ReTrms:
    """Construct random effects terms from formula and data.

    This function parses the formula, extracts the random effects
    specification, and constructs the design matrices and parameter
    vectors needed for fitting a mixed model.

    Parameters
    ----------
    formula : str
        Model formula with random effects (e.g., "y ~ x + (1|group)").
    data : pd.DataFrame
        Data frame containing the variables in the formula.

    Returns
    -------
    ReTrms
        Structure containing random effects terms.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'y': [1, 2, 3, 4, 5, 6],
    ...     'x': [1, 2, 1, 2, 1, 2],
    ...     'group': ['A', 'A', 'B', 'B', 'C', 'C']
    ... })
    >>> re_terms = mkReTrms("y ~ x + (1|group)", data)
    >>> re_terms.Zt.shape
    (3, 6)
    >>> re_terms.nl
    [3]

    See Also
    --------
    lmer : Fit linear mixed models.
    glmer : Fit generalized linear mixed models.
    lFormula : Parse formula for LMM.
    """
    from scipy import sparse

    parsed_formula = parse_formula(formula)
    matrices = build_model_matrices(parsed_formula, data)

    theta = []
    Lind = []
    Gp = [0]
    flist = {}
    cnms = {}
    nl = []

    current_idx = 0
    theta_idx = 0

    for struct in matrices.random_structures:
        group_name = struct.grouping_factor
        n_levels = struct.n_levels
        n_terms = struct.n_terms

        levels = sorted(struct.level_map.keys(), key=lambda x: struct.level_map[x])
        flist[group_name] = np.array(levels)
        cnms[group_name] = struct.term_names
        nl.append(n_levels)

        if struct.correlated:
            for i in range(n_terms):
                for j in range(i + 1):
                    if i == j:
                        theta.append(1.0)
                    else:
                        theta.append(0.0)
                    for _ in range(n_levels):
                        Lind.append(theta_idx)
                    theta_idx += 1
        else:
            for _ in range(n_terms):
                theta.append(1.0)
                for _ in range(n_levels):
                    Lind.append(theta_idx)
                theta_idx += 1

        current_idx += n_levels * n_terms
        Gp.append(current_idx)

    Zt = matrices.Z.T.tocsc() if sparse.issparse(matrices.Z) else sparse.csc_matrix(matrices.Z.T)

    return ReTrms(
        Zt=Zt,
        theta=np.array(theta),
        Lind=np.array(Lind),
        Gp=Gp,
        flist=flist,
        cnms=cnms,
        nl=nl,
        _formula=parsed_formula,
        _matrices=matrices,
    )


def simulate_formula(
    formula: Formula | str,
    data: pd.DataFrame,
    beta: NDArray[np.floating] | dict[str, float] | None = None,
    theta: NDArray[np.floating] | None = None,
    sigma: float = 1.0,
    family: Family | str | None = None,
    nsim: int = 1,
    seed: int | None = None,
) -> pd.DataFrame | list[pd.DataFrame]:
    """Simulate response data from a formula before fitting.

    This function generates simulated response data based on a formula,
    fixed effects, and variance components, without first fitting a model.
    This is useful for power analysis, simulation studies, and understanding
    model behavior.

    Parameters
    ----------
    formula : Formula or str
        Parsed formula or model formula with random effects (e.g., "y ~ x + (1|group)").
    data : pd.DataFrame
        Data frame containing the predictor variables. The response column
        may be absent; a grouped binomial ``successes / trials`` response
        needs the trials column and receives simulated success counts.
    beta : array-like or dict, optional
        Fixed effects coefficients. If dict, keys must be coefficient names;
        omitted coefficients are zero. If None, uses zeros.
    theta : array-like, optional
        Variance component parameters (relative covariance factors).
        If None, uses ones.
    sigma : float, default 1.0
        Positive residual scale, which also scales the random effects.
        Gaussian responses have standard deviation ``sigma``; gamma and
        inverse Gaussian responses have dispersion ``sigma**2``, that is,
        shape ``1 / sigma**2``. Binomial, Poisson and negative binomial draws
        do not use it.
    family : Family or str, optional
        Distribution family with a ``simulate()`` method, or one of
        "gaussian", "binomial", "poisson", "gamma" and "inverse_gaussian"
        (R spellings such as "Gamma" and "inverse.gaussian" also work).
        If None, uses Gaussian.
    nsim : int, default 1
        Number of simulations to generate.
    seed : int, optional
        Random seed for reproducibility. Uses an isolated generator and does
        not alter NumPy's global random state.

    Returns
    -------
    DataFrame or list of DataFrame
        If nsim=1, returns a single DataFrame with simulated response.
        If nsim>1, returns a list of DataFrames.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = pd.DataFrame({
    ...     'x': np.random.randn(100),
    ...     'group': np.repeat(['A', 'B', 'C', 'D', 'E'], 20)
    ... })
    >>> # Simulate with specific effects
    >>> beta = {'(Intercept)': 5.0, 'x': 2.0}
    >>> simulated = simulate_formula(
    ...     "y ~ x + (1|group)",
    ...     data,
    ...     beta=beta,
    ...     theta=[0.5],
    ...     sigma=1.0
    ... )
    >>> simulated['y'].mean()  # Should be around 5

    >>> # Multiple simulations for power analysis
    >>> sims = simulate_formula(
    ...     "y ~ x + (1|group)",
    ...     data,
    ...     beta={'(Intercept)': 0, 'x': 0.5},
    ...     nsim=100,
    ...     seed=42
    ... )

    See Also
    --------
    lmer : Fit linear mixed models.
    LmerResult.simulate : Simulate from a fitted model.
    """
    from mixedlm.families import Binomial
    from mixedlm.utils.simulation import simulate_glmm_response

    if nsim < 1:
        raise ValueError("nsim must be at least 1")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")

    family = _simulation_family(family)
    rng = np.random.default_rng(seed)

    parsed_formula = _coerce_formula(formula)
    response_name = parsed_formula.response
    # The response is only an output here, so the predictors suffice.
    frame = data if response_name in data.columns else data.assign(**{response_name: 0.0})
    matrices = build_model_matrices(
        parsed_formula, frame, grouped_binomial=isinstance(family, Binomial)
    )

    p = matrices.n_fixed
    q = matrices.n_random

    if beta is None:
        beta_vec: NDArray[np.floating] = np.zeros(p)
    elif isinstance(beta, dict):
        unknown = sorted(set(beta) - set(matrices.fixed_names))
        if unknown:
            raise ValueError(
                f"beta has unknown coefficient names {unknown}; available: {matrices.fixed_names}"
            )
        beta_vec = np.array([beta.get(name, 0.0) for name in matrices.fixed_names], dtype=float)
    else:
        beta_vec = np.asarray(beta, dtype=np.float64).reshape(-1)

    if len(beta_vec) != p:
        raise ValueError(f"beta has length {len(beta_vec)}; expected {p}")

    if theta is None:
        theta_values: list[float] = []
        for struct in matrices.random_structures:
            cov_type = getattr(struct, "cov_type", "us")
            if cov_type in ("cs", "ar1"):
                theta_values.append(1.0)
                if struct.n_terms > 1:
                    theta_values.append(0.0)
            else:
                n_struct_theta = (
                    struct.n_terms * (struct.n_terms + 1) // 2
                    if struct.correlated
                    else struct.n_terms
                )
                theta_values.extend([1.0] * n_struct_theta)
        theta_vec: NDArray[np.floating] = np.asarray(theta_values, dtype=np.float64)
    else:
        theta_vec = np.asarray(theta, dtype=np.float64).reshape(-1)

    n_theta = _count_theta(matrices.random_structures)
    if theta_vec.size != n_theta:
        raise ValueError(f"theta must contain {n_theta} parameters, got {theta_vec.size}")
    if not np.all(np.isfinite(theta_vec)):
        raise ValueError("theta must contain only finite values")

    Lambda = _build_lambda(theta_vec, matrices.random_structures)
    precision = matrices.weights / sigma**2
    results = []

    for _ in range(nsim):
        eta = matrices.X @ beta_vec
        standard_random_effects = rng.standard_normal(q)
        random_effects = np.asarray(Lambda @ standard_random_effects).reshape(-1) * sigma
        eta += matrices.Z @ random_effects

        y = simulate_glmm_response(
            family, family.link.inverse(eta), precision, trials=matrices.trials, rng=rng
        )
        result_df = data.copy()
        result_df[response_name] = y
        results.append(result_df)

    if nsim == 1:
        return results[0]
    return results


def _simulation_family(family: Family | str | None) -> Family:
    """Return the family for a Family instance, None (Gaussian) or an R family name."""
    from mixedlm import families

    if family is None:
        return families.Gaussian()
    if not isinstance(family, str):
        return family
    constructors: dict[str, Callable[[], Family]] = {
        "gaussian": families.Gaussian,
        "binomial": families.Binomial,
        "poisson": families.Poisson,
        "gamma": families.Gamma,
        "inverse_gaussian": families.InverseGaussian,
    }
    name = family.strip().lower().replace(".", "_")
    if name not in constructors:
        raise ValueError(
            f"Unknown family {family!r}; choose a Family instance or one of: "
            + ", ".join(constructors)
        )
    return constructors[name]()


def _template_variables(formula: Formula) -> tuple[list[str], list[str]]:
    """Return a formula's covariates and grouping factors in order of appearance."""
    grouping_factors = list(
        dict.fromkeys(g for rterm in formula.random for g in rterm.grouping_factors)
    )
    covariates: dict[str, None] = {}
    random_terms = [term for rterm in formula.random for term in rterm.expr]
    for term in [*formula.fixed.terms, *random_terms]:
        if isinstance(term, VariableTerm | PowerTerm):
            covariates[term.name] = None
        elif isinstance(term, InteractionTerm):
            covariates.update(dict.fromkeys(term.source_variables))
    return [name for name in covariates if name not in grouping_factors], grouping_factors


def mkDataTemplate(
    formula: Formula | str,
    nlevs: dict[str, int] | None = None,
    balanced: bool = True,
) -> pd.DataFrame:
    """Create a template data frame for a mixed model formula.

    This function generates a data frame with the structure implied by
    a model formula, useful for simulation studies and power analysis.

    Parameters
    ----------
    formula : Formula or str
        Model formula with random effects (e.g., "y ~ x + (1|group)").
    nlevs : dict, optional
        Dictionary mapping grouping factor names to number of levels.
        Unlisted factors have 10 levels.
    balanced : bool, default True
        If True, creates one row for each combination of grouping-factor
        levels. If False, creates twice as many rows as there are levels in
        total; each level has at least one row and the remaining rows are
        assigned to random levels.

    Returns
    -------
    pd.DataFrame
        The response, then standard normal covariates, then grouping factors
        with levels named ``<factor>1``, ``<factor>2``, and so on.

    Examples
    --------
    >>> df = mkDataTemplate("y ~ x + (1|subject)", nlevs={"subject": 20})
    >>> df.shape
    (20, 3)

    >>> df = mkDataTemplate(
    ...     "y ~ x + (1|subject) + (1|item)",
    ...     nlevs={"subject": 10, "item": 5}
    ... )
    >>> df.shape
    (50, 4)
    """
    parsed = _coerce_formula(formula)
    covariates, grouping_factors = _template_variables(parsed)
    level_counts = [(nlevs or {}).get(g, 10) for g in grouping_factors]
    labels = [
        np.array([f"{g}{i + 1}" for i in range(count)])
        for g, count in zip(grouping_factors, level_counts, strict=True)
    ]
    n = int(np.prod(level_counts)) if balanced else 2 * sum(level_counts)

    data: dict[str, object] = {parsed.response: np.random.randn(n)}
    for var in covariates:
        data[var] = np.random.randn(n)

    if balanced:
        # One row for each combination of grouping-factor levels.
        grids = np.meshgrid(*labels, indexing="ij")
        for g, grid in zip(grouping_factors, grids, strict=True):
            data[g] = grid.ravel()
    else:
        for g, levels in zip(grouping_factors, labels, strict=True):
            # Every level gets one row; the remaining rows go to random levels.
            extra = np.random.randint(len(levels), size=n - len(levels))
            data[g] = levels[np.sort(np.concatenate([np.arange(len(levels)), extra]))]

    return pd.DataFrame(data)


def mkParsTemplate(
    formula: Formula | str,
    data: pd.DataFrame,
) -> dict[str, object]:
    """Generate a parameter structure template from formula and data.

    This function creates a template dictionary showing the parameter
    structure implied by a model formula, including fixed effects
    names and variance component structure.

    Parameters
    ----------
    formula : Formula or str
        Model formula with random effects.
    data : pd.DataFrame
        Data frame containing the variables.

    Returns
    -------
    dict
        Dictionary with:
        - 'beta': dict of fixed effect names with None placeholders
        - 'theta': one label per variance parameter, in theta order:
          ``sd_<term>|<group>`` and ``cor_<term>_<term>|<group>`` for the
          diagonal and off-diagonal relative Cholesky entries, or
          ``sd|<group>`` and ``rho|<group>`` for compound-symmetry and AR(1)
          scale and correlation
        - 'sigma': placeholder for residual SD
        - 'n_fixed': number of fixed effects
        - 'n_theta': number of variance parameters

    Examples
    --------
    >>> data = pd.DataFrame({'y': [1,2,3], 'x': [1,2,3], 'g': ['A','B','A']})
    >>> template = mkParsTemplate("y ~ x + (1|g)", data)
    >>> template['beta']
    {'(Intercept)': None, 'x': None}
    >>> template['n_theta']
    1
    """
    parsed = lFormula(formula, data)

    beta_template = {name: None for name in parsed.matrices.fixed_names}

    theta_template = []
    for struct in parsed.matrices.random_structures:
        group = struct.grouping_factor
        terms = struct.term_names
        q = struct.n_terms

        if struct.cov_type in ("cs", "ar1"):
            # A common relative scale, then one correlation for several terms.
            theta_template.append(f"sd|{group}")
            if q > 1:
                theta_template.append(f"rho|{group}")
        elif struct.correlated:
            for i in range(q):
                for j in range(i + 1):
                    if i == j:
                        theta_template.append(f"sd_{terms[i]}|{group}")
                    else:
                        theta_template.append(f"cor_{terms[j]}_{terms[i]}|{group}")
        else:
            for term in terms:
                theta_template.append(f"sd_{term}|{group}")

    return {
        "beta": beta_template,
        "theta": theta_template,
        "sigma": None,
        "n_fixed": len(beta_template),
        "n_theta": len(theta_template),
    }


def mkMinimalData(
    formula: Formula | str,
    n: int = 10,
) -> pd.DataFrame:
    """Create minimal test data from a formula.

    This function generates a minimal data frame suitable for testing
    that a formula can be parsed and model matrices can be built.

    Parameters
    ----------
    formula : Formula or str
        Model formula with random effects.
    n : int, default 10
        Number of observations.

    Returns
    -------
    pd.DataFrame
        The response, then standard normal covariates in formula order, then
        grouping factors with ``min(n, 5)`` levels assigned cyclically.

    Examples
    --------
    >>> df = mkMinimalData("y ~ x + z + (1|group)")
    >>> list(df.columns)
    ['y', 'x', 'z', 'group']
    """
    parsed = _coerce_formula(formula)
    covariates, grouping_factors = _template_variables(parsed)

    data: dict[str, object] = {parsed.response: np.random.randn(n)}
    for var in covariates:
        data[var] = np.random.randn(n)

    n_levels = min(n, 5)
    for g in grouping_factors:
        data[g] = [f"{g}{i % n_levels + 1}" for i in range(n)]

    return pd.DataFrame(data)


def mkNewReTrms(
    reTrms: ReTrms,
    newdata: pd.DataFrame,
) -> ReTrms:
    """Create new random effect terms structure from existing one and new data.

    This function constructs random effect design matrices for new data
    using the structure from an existing ReTrms object. This is useful
    for prediction with new groups or new observations.

    Parameters
    ----------
    reTrms : ReTrms
        Random effect terms from mkReTrms.
    newdata : pd.DataFrame
        New data with the random-effect covariates and grouping factors.

    Returns
    -------
    ReTrms
        Random effect terms for the new data. Columns of ``Zt`` keep the
        coefficient order of ``reTrms``.

    Notes
    -----
    Factor covariates are encoded with the original levels and contrasts.
    Rows whose grouping-factor level is not in ``reTrms`` have no entries in
    that term's columns.

    Examples
    --------
    >>> reTrms = mkReTrms("y ~ x + (1|group)", train_data)
    >>> new_reTrms = mkNewReTrms(reTrms, test_data)
    """
    from scipy import sparse

    formula, source = reTrms._formula, reTrms._matrices
    if formula is None or source is None:
        raise ValueError("reTrms must be created by mkReTrms")

    Z, structures = build_random_matrix(
        formula, newdata, contrasts=source.contrasts, category_levels=source.category_levels
    )
    Z = Z.tocoo()
    columns = np.full(Z.nnz, -1, dtype=np.int64)
    new_offset = source_offset = 0
    for fitted, new in zip(source.random_structures, structures, strict=True):
        if new.term_names != fitted.term_names:
            raise ValueError(
                f"newdata produced different random-effect columns for '{fitted.grouping_factor}'"
            )
        new_levels = sorted(new.level_map, key=new.level_map.__getitem__)
        source_levels = np.array([fitted.level_map.get(level, -1) for level in new_levels])
        in_block = (Z.col >= new_offset) & (Z.col < new_offset + new.n_levels * new.n_terms)
        level, term = np.divmod(Z.col[in_block] - new_offset, new.n_terms)
        target = source_levels[level]
        columns[in_block] = np.where(
            target >= 0, source_offset + target * fitted.n_terms + term, -1
        )
        new_offset += new.n_levels * new.n_terms
        source_offset += fitted.n_levels * fitted.n_terms

    known = columns >= 0
    Zt = sparse.csc_matrix(
        (Z.data[known], (columns[known], Z.row[known])),
        shape=(source.n_random, Z.shape[0]),
    )
    return replace(
        reTrms,
        Zt=Zt,
        theta=reTrms.theta.copy(),
        Lind=reTrms.Lind.copy(),
        Gp=list(reTrms.Gp),
        flist=dict(reTrms.flist),
        cnms=dict(reTrms.cnms),
        nl=list(reTrms.nl),
    )


def devfun2(
    devfun: LmerDevfun | GlmerDevfun,
    theta_opt: NDArray[np.floating],
    which: int | list[int] | None = None,
) -> Callable[[NDArray[np.floating]], float]:
    """Create a stripped deviance function for profiling.

    This function creates a simplified deviance function that can be
    used for profile likelihood calculations. It holds some parameters
    fixed at their optimal values while allowing others to vary.

    Parameters
    ----------
    devfun : LmerDevfun or GlmerDevfun
        The original deviance function.
    theta_opt : ndarray
        The optimal theta values from the fitted model.
    which : int or list of int, optional
        Which theta parameters to allow to vary. If None, all vary.

    Returns
    -------
    callable
        A deviance function suitable for profiling.

    Examples
    --------
    >>> from mixedlm import lmer
    >>> result = lmer("y ~ x + (1|group)", data)
    >>> parsed = lFormula("y ~ x + (1|group)", data)
    >>> devfun = mkLmerDevfun(parsed)
    >>> # Profile the first variance component
    >>> profile_devfun = devfun2(devfun, result.theta, which=[0])

    See Also
    --------
    profile_lmer : Profile likelihood for LMMs.
    confint : Confidence intervals including profile method.
    """
    theta_opt = np.asarray(theta_opt)

    if which is None:
        which_list = list(range(len(theta_opt)))
    elif isinstance(which, int):
        which_list = [which]
    else:
        which_list = list(which)

    def profiled_devfun(theta_partial: NDArray[np.floating]) -> float:
        theta_full = theta_opt.copy()
        for i, w in enumerate(which_list):
            if i < len(theta_partial):
                theta_full[w] = theta_partial[i]
        return devfun(theta_full)

    return profiled_devfun
