from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pandas as pd

    from mixedlm.estimation.laplace import GLMMOptimizer
    from mixedlm.families.base import Family
    from mixedlm.formula.terms import Formula
    from mixedlm.models.control import GlmerControl, LmerControl
    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult

from mixedlm.estimation.reml import (
    LMMOptimizer,
    _build_theta_bounds,
    _count_theta,
)
from mixedlm.formula.parser import parse_formula
from mixedlm.matrices.design import ModelMatrices, build_model_matrices


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

    This class wraps the Laplace approximation deviance function
    and provides methods for evaluation and optimization.

    Attributes
    ----------
    parsed : GlmerParsedFormula
        The parsed formula result from glFormula.
    optimizer : GLMMOptimizer
        The optimizer object for computing deviance.
    """

    parsed: GlmerParsedFormula
    optimizer: GLMMOptimizer

    def __call__(self, theta: NDArray[np.floating]) -> float:
        """Evaluate the deviance function at theta.

        Parameters
        ----------
        theta : NDArray
            Variance component parameters (relative covariance factors).

        Returns
        -------
        float
            The Laplace approximation to the deviance.
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

        Returns
        -------
        list of tuple
            Bounds for each theta parameter.
        """
        return _build_theta_bounds(self.parsed.matrices.random_structures, self.parsed.n_theta)


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
    """

    theta: NDArray[np.floating]
    deviance: float
    converged: bool
    n_iter: int
    message: str


def lFormula(
    formula: str,
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
    formula : str
        Model formula in lme4 syntax (e.g., "y ~ x + (1|group)").
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
    parsed_formula = parse_formula(formula)
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
    formula: str,
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
    formula : str
        Model formula in lme4 syntax (e.g., "y ~ x + (1|group)").
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

    parsed_formula = parse_formula(formula)
    matrices = build_model_matrices(
        parsed_formula,
        data,
        weights=weights,
        offset=offset,
        na_action=na_action,
        contrasts=contrasts,
    )

    if family is None:
        family = Binomial()

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

    return LmerDevfun(parsed=parsed, optimizer=optimizer)


def mkGlmerDevfun(
    parsed: GlmerParsedFormula,
    verbose: int = 0,
    control: GlmerControl | None = None,
) -> GlmerDevfun:
    """Create the deviance function for a generalized linear mixed model.

    This is the second step in the modular interface for GLMMs. It creates
    the objective function (Laplace approximation) that will be minimized.

    Parameters
    ----------
    parsed : GlmerParsedFormula
        Result from glFormula.
    verbose : int, default 0
        Verbosity level for optimization output.
    control : GlmerControl, optional
        Control parameters for the optimizer.

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
    )

    return GlmerDevfun(parsed=parsed, optimizer=optimizer)


def optimizeLmer(
    devfun: LmerDevfun,
    start: NDArray[np.floating] | None = None,
    method: str = "L-BFGS-B",
    maxiter: int = 1000,
    verbose: int = 0,
) -> OptimizeResult:
    """Optimize the deviance function for a linear mixed model.

    This is the third step in the modular interface. It minimizes
    the deviance function to find optimal variance components.

    Parameters
    ----------
    devfun : LmerDevfun
        Deviance function from mkLmerDevfun.
    start : NDArray, optional
        Starting values for theta. If None, uses default.
    method : str, default "L-BFGS-B"
        Optimization method (passed to scipy.optimize.minimize).
    maxiter : int, default 1000
        Maximum number of iterations.
    verbose : int, default 0
        Verbosity level.

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
    from scipy.optimize import minimize

    if start is None:
        start = devfun.get_start()

    bounds = devfun.get_bounds()

    callback: Callable[[NDArray[np.floating]], None] | None = None
    if verbose > 0:

        def callback(x: NDArray[np.floating]) -> None:
            dev = devfun(x)
            print(f"theta = {x}, deviance = {dev:.6f}")

    result = minimize(
        devfun,
        start,
        method=method,
        bounds=bounds,
        options={"maxiter": maxiter},
        callback=callback,
    )

    return OptimizeResult(
        theta=result.x,
        deviance=result.fun,
        converged=result.success,
        n_iter=result.nit,
        message=result.message if hasattr(result, "message") else "",
    )


def optimizeGlmer(
    devfun: GlmerDevfun,
    start: NDArray[np.floating] | None = None,
    method: str = "L-BFGS-B",
    maxiter: int = 1000,
    verbose: int = 0,
) -> OptimizeResult:
    """Optimize the deviance function for a generalized linear mixed model.

    This is the third step in the modular interface for GLMMs.

    Parameters
    ----------
    devfun : GlmerDevfun
        Deviance function from mkGlmerDevfun.
    start : NDArray, optional
        Starting values for theta. If None, uses default.
    method : str, default "L-BFGS-B"
        Optimization method (passed to scipy.optimize.minimize).
    maxiter : int, default 1000
        Maximum number of iterations.
    verbose : int, default 0
        Verbosity level.

    Returns
    -------
    OptimizeResult
        Optimization result containing theta, deviance, and convergence info.

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
    from scipy.optimize import minimize

    if start is None:
        start = devfun.get_start()

    bounds = devfun.get_bounds()

    callback: Callable[[NDArray[np.floating]], None] | None = None
    if verbose > 0:

        def callback(x: NDArray[np.floating]) -> None:
            dev = devfun(x)
            print(f"theta = {x}, deviance = {dev:.6f}")

    result = minimize(
        devfun,
        start,
        method=method,
        bounds=bounds,
        options={"maxiter": maxiter},
        callback=callback,
    )

    return OptimizeResult(
        theta=result.x,
        deviance=result.fun,
        converged=result.success,
        n_iter=result.nit,
        message=result.message if hasattr(result, "message") else "",
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

    beta, sigma, u = devfun.optimizer._extract_estimates(opt.theta)

    return LmerResult(
        formula=devfun.parsed.formula,
        matrices=devfun.parsed.matrices,
        theta=opt.theta,
        beta=beta,
        sigma=sigma,
        u=u,
        deviance=opt.deviance,
        REML=devfun.parsed.REML,
        converged=opt.converged,
        n_iter=opt.n_iter,
    )


def mkGlmerMod(
    devfun: GlmerDevfun,
    opt: OptimizeResult,
    nAGQ: int = 1,
) -> GlmerResult:
    """Create a GlmerResult from optimization results.

    This is the final step in the modular interface for GLMMs.

    Parameters
    ----------
    devfun : GlmerDevfun
        Deviance function from mkGlmerDevfun.
    opt : OptimizeResult
        Optimization result from optimizeGlmer.
    nAGQ : int, default 1
        Number of adaptive Gauss-Hermite quadrature points used.

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
    from mixedlm.estimation.laplace import laplace_deviance
    from mixedlm.models.glmer import GlmerResult

    _, beta, u = laplace_deviance(
        opt.theta,
        devfun.parsed.matrices,
        devfun.parsed.family,
        devfun.optimizer._beta_cache,
        devfun.optimizer._u_cache,
    )

    return GlmerResult(
        formula=devfun.parsed.formula,
        matrices=devfun.parsed.matrices,
        family=devfun.parsed.family,
        theta=opt.theta,
        beta=beta,
        u=u,
        deviance=opt.deviance,
        converged=opt.converged,
        n_iter=opt.n_iter,
        nAGQ=nAGQ,
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
    )


def simulate_formula(
    formula: str,
    data: pd.DataFrame,
    beta: NDArray[np.floating] | dict[str, float] | None = None,
    theta: NDArray[np.floating] | None = None,
    sigma: float = 1.0,
    family: Family | None = None,
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
    formula : str
        Model formula with random effects (e.g., "y ~ x + (1|group)").
    data : pd.DataFrame
        Data frame containing the predictor variables.
    beta : array-like or dict, optional
        Fixed effects coefficients. If dict, keys should be coefficient
        names. If None, uses zeros.
    theta : array-like, optional
        Variance component parameters (relative covariance factors).
        If None, uses ones.
    sigma : float, default 1.0
        Residual standard deviation (for Gaussian family).
    family : Family, optional
        Distribution family. If None, uses Gaussian.
    nsim : int, default 1
        Number of simulations to generate.
    seed : int, optional
        Random seed for reproducibility.

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
    from scipy import sparse

    from mixedlm.families import Gaussian

    if seed is not None:
        np.random.seed(seed)

    if family is None:
        family = Gaussian()

    parsed_formula = parse_formula(formula)
    matrices = build_model_matrices(parsed_formula, data)

    n = matrices.n_obs
    p = matrices.n_fixed
    q = matrices.n_random

    if beta is None:
        beta_vec = np.zeros(p)
    elif isinstance(beta, dict):
        beta_vec = np.zeros(p)
        for i, name in enumerate(matrices.fixed_names):
            if name in beta:
                beta_vec[i] = beta[name]
    else:
        beta_vec = np.asarray(beta)

    if theta is None:
        n_theta = sum(
            s.n_terms * (s.n_terms + 1) // 2 if s.correlated else s.n_terms
            for s in matrices.random_structures
        )
        theta_vec = np.ones(n_theta)
    else:
        theta_vec = np.asarray(theta)

    results = []

    for _ in range(nsim):
        eta = matrices.X @ beta_vec

        u = np.zeros(q)
        idx = 0
        theta_idx = 0

        for struct in matrices.random_structures:
            n_levels = struct.n_levels
            n_terms = struct.n_terms

            if struct.correlated:
                n_theta_block = n_terms * (n_terms + 1) // 2
                theta_block = theta_vec[theta_idx : theta_idx + n_theta_block]
                theta_idx += n_theta_block

                L = np.zeros((n_terms, n_terms))
                k = 0
                for j in range(n_terms):
                    for i in range(j, n_terms):
                        L[i, j] = theta_block[k]
                        k += 1

                for _level in range(n_levels):
                    z = np.random.randn(n_terms)
                    u_level = L @ z * sigma
                    u[idx : idx + n_terms] = u_level
                    idx += n_terms
            else:
                theta_block = theta_vec[theta_idx : theta_idx + n_terms]
                theta_idx += n_terms

                for term in range(n_terms):
                    for _level in range(n_levels):
                        u[idx] = np.random.randn() * theta_block[term] * sigma
                        idx += 1

        if sparse.issparse(matrices.Z):
            eta += matrices.Z @ u
        else:
            eta += matrices.Z @ u

        mu = family.link.inverse(eta)

        family_name = getattr(family, "name", "gaussian")
        if family_name == "gaussian":
            y = mu + np.random.randn(n) * sigma
        elif family_name == "binomial":
            y = np.random.binomial(1, np.clip(mu, 0, 1)).astype(float)
        elif family_name == "poisson":
            y = np.random.poisson(np.maximum(mu, 0)).astype(float)
        else:
            y = mu + np.random.randn(n) * sigma

        result_df = data.copy()
        response_name = parsed_formula.response if parsed_formula.response else "y"
        result_df[response_name] = y
        results.append(result_df)

    if nsim == 1:
        return results[0]
    return results


def mkDataTemplate(
    formula: str,
    nlevs: dict[str, int] | None = None,
    balanced: bool = True,
) -> pd.DataFrame:
    """Create a template data frame for a mixed model formula.

    This function generates a data frame with the structure implied by
    a model formula, useful for simulation studies and power analysis.

    Parameters
    ----------
    formula : str
        Model formula with random effects (e.g., "y ~ x + (1|group)").
    nlevs : dict, optional
        Dictionary mapping grouping factor names to number of levels.
        If not specified, uses 10 levels per factor.
    balanced : bool, default True
        If True, creates a balanced design. If False, creates an
        unbalanced design with varying observations per group.

    Returns
    -------
    pd.DataFrame
        Template data frame with appropriate structure.

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
    import re

    bar_pattern = r"\(([^|]+)\|([^)]+)\)"
    matches = re.findall(bar_pattern, formula)

    grouping_factors = [match[1].strip() for match in matches]

    if nlevs is None:
        nlevs = {g: 10 for g in grouping_factors}

    lhs_rhs = formula.split("~")
    response = lhs_rhs[0].strip()

    rhs = lhs_rhs[1] if len(lhs_rhs) > 1 else ""
    rhs_no_bars = re.sub(bar_pattern, "", rhs)
    rhs_no_bars = re.sub(r"\s*\+\s*\+\s*", " + ", rhs_no_bars)
    rhs_no_bars = rhs_no_bars.strip(" +")

    fixed_vars = []
    if rhs_no_bars:
        terms = [t.strip() for t in rhs_no_bars.split("+")]
        for term in terms:
            if term and term != "1" and term != "0":
                fixed_vars.append(term)

    if balanced:
        total_levels = 1
        for g in grouping_factors:
            total_levels *= nlevs.get(g, 10)
        n = total_levels
    else:
        n = sum(nlevs.get(g, 10) for g in grouping_factors) * 2

    data: dict[str, object] = {response: np.random.randn(n)}

    for var in fixed_vars:
        data[var] = np.random.randn(n)

    if balanced and len(grouping_factors) >= 2:
        levels_list = [list(range(1, nlevs.get(g, 10) + 1)) for g in grouping_factors]
        grids = np.meshgrid(*levels_list, indexing="ij")
        for i, g in enumerate(grouping_factors):
            data[g] = [f"{g}{v}" for v in grids[i].ravel()]
    else:
        for g in grouping_factors:
            n_levels = nlevs.get(g, 10)
            if balanced:
                obs_per_level = max(1, n // n_levels)
                levels = [f"{g}{i + 1}" for i in range(n_levels) for _ in range(obs_per_level)]
                data[g] = levels[:n]
            else:
                obs_per_level_arr = np.random.randint(1, 5, n_levels)
                levels = []
                for i, count in enumerate(obs_per_level_arr.tolist()):
                    levels.extend([f"{g}{i + 1}"] * count)
                if len(levels) < n:
                    levels.extend(levels[: n - len(levels)])
                else:
                    levels = levels[:n]
                data[g] = levels

    return pd.DataFrame(data)


def mkParsTemplate(
    formula: str,
    data: pd.DataFrame,
) -> dict[str, object]:
    """Generate a parameter structure template from formula and data.

    This function creates a template dictionary showing the parameter
    structure implied by a model formula, including fixed effects
    names and variance component structure.

    Parameters
    ----------
    formula : str
        Model formula with random effects.
    data : pd.DataFrame
        Data frame containing the variables.

    Returns
    -------
    dict
        Dictionary with:
        - 'beta': dict of fixed effect names with None placeholders
        - 'theta': list of theta parameter descriptions
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

        if struct.correlated:
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
    formula: str,
    n: int = 10,
) -> pd.DataFrame:
    """Create minimal test data from a formula.

    This function generates a minimal data frame suitable for testing
    that a formula can be parsed and model matrices can be built.

    Parameters
    ----------
    formula : str
        Model formula with random effects.
    n : int, default 10
        Number of observations.

    Returns
    -------
    pd.DataFrame
        Minimal data frame with required variables.

    Examples
    --------
    >>> df = mkMinimalData("y ~ x + z + (1|group)")
    >>> list(df.columns)
    ['y', 'x', 'z', 'group']
    """
    import re

    lhs_rhs = formula.split("~")
    response = lhs_rhs[0].strip()

    bar_pattern = r"\(([^|]+)\|([^)]+)\)"
    matches = re.findall(bar_pattern, formula)
    grouping_factors = list(set(match[1].strip() for match in matches))

    re_terms = set()
    for terms_str, _ in matches:
        for term in terms_str.split("+"):
            term = term.strip()
            if term and term != "1" and term != "0":
                re_terms.add(term)

    rhs = lhs_rhs[1] if len(lhs_rhs) > 1 else ""
    rhs_no_bars = re.sub(bar_pattern, "", rhs)
    rhs_no_bars = re.sub(r"\s*\+\s*\+\s*", " + ", rhs_no_bars)
    rhs_no_bars = rhs_no_bars.strip(" +")

    fixed_vars = set()
    if rhs_no_bars:
        for term in rhs_no_bars.split("+"):
            term = term.strip()
            if term and term != "1" and term != "0" and ":" not in term and "*" not in term:
                fixed_vars.add(term)

    all_numeric = fixed_vars | re_terms

    data: dict[str, object] = {response: np.random.randn(n)}

    for var in all_numeric:
        data[var] = np.random.randn(n)

    for g in grouping_factors:
        n_levels = min(n, 5)
        levels = [f"{g}{i % n_levels + 1}" for i in range(n)]
        data[g] = levels

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
        Existing random effect terms from a fitted model.
    newdata : pd.DataFrame
        New data for which to create design matrices.

    Returns
    -------
    ReTrms
        New random effect terms for the new data.

    Notes
    -----
    If new data contains group levels not in the original data, the
    corresponding rows in Z will be all zeros unless allow_new_levels
    is handled by the caller.

    Examples
    --------
    >>> # Fit a model and get ReTrms
    >>> reTrms = mkReTrms("y ~ x + (1|group)", train_data)
    >>> # Create ReTrms for new data
    >>> new_reTrms = mkNewReTrms(reTrms, test_data)
    """
    from scipy import sparse

    n_new = len(newdata)
    total_re = sum(len(levels) * len(reTrms.cnms[g]) for g, levels in reTrms.flist.items())

    Z_rows = []
    Z_cols = []
    Z_data = []

    col_offset = 0
    new_flist: dict[str, NDArray] = {}
    new_nl: list[int] = []

    for group_name in reTrms.flist:
        original_levels = list(reTrms.flist[group_name])
        term_names = reTrms.cnms[group_name]
        n_terms = len(term_names)
        n_levels = len(original_levels)

        level_map = {lvl: i for i, lvl in enumerate(original_levels)}

        new_flist[group_name] = np.array(original_levels)
        new_nl.append(n_levels)

        if group_name not in newdata.columns:
            col_offset += n_levels * n_terms
            continue

        group_col = newdata[group_name].values

        for row_idx, level in enumerate(group_col):
            if level in level_map:
                level_idx = level_map[level]
                for t, term in enumerate(term_names):
                    col_idx = col_offset + level_idx * n_terms + t
                    if term == "(Intercept)" or term == "1":
                        val = 1.0
                    elif term in newdata.columns:
                        val = float(newdata[term].iloc[row_idx])
                    else:
                        val = 1.0

                    Z_rows.append(row_idx)
                    Z_cols.append(col_idx)
                    Z_data.append(val)

        col_offset += n_levels * n_terms

    if Z_rows:
        Z = sparse.csr_matrix(
            (Z_data, (Z_rows, Z_cols)),
            shape=(n_new, total_re),
            dtype=np.float64,
        )
    else:
        Z = sparse.csr_matrix((n_new, total_re), dtype=np.float64)

    Zt = Z.T.tocsc()

    return ReTrms(
        Zt=Zt,
        theta=reTrms.theta.copy(),
        Lind=reTrms.Lind.copy(),
        Gp=list(reTrms.Gp),
        flist=new_flist,
        cnms=dict(reTrms.cnms),
        nl=new_nl,
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
