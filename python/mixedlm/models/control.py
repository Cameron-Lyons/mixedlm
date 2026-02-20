from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_VALID_OPTIMIZERS = {
    "L-BFGS-B",
    "BFGS",
    "Nelder-Mead",
    "Powell",
    "trust-constr",
    "SLSQP",
    "TNC",
    "COBYLA",
    "bobyqa",
    "nloptwrap_BOBYQA",
    "nloptwrap_NEWUOA",
    "nloptwrap_PRAXIS",
    "nloptwrap_SBPLX",
    "nloptwrap_COBYLA",
    "nloptwrap_NELDERMEAD",
}

_VALID_CHECK_ACTIONS = {"ignore", "warning", "message", "stop"}
_VALID_RANKX_ACTIONS = {
    "ignore",
    "warning",
    "message",
    "stop",
    "message+drop.cols",
    "warning+drop.cols",
    "stop+drop.cols",
}


def _validate_common_control(
    *,
    optimizer: str,
    maxiter: int,
    boundary_tol: float,
    check_nobs_vs_rankZ: str,
    check_nobs_vs_nlev: str,
    check_nlev_gtreq_5: str,
    check_nlev_gtr_1: str,
    check_scaleX: str,
    check_rankX: str,
) -> None:
    if optimizer not in _VALID_OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer '{optimizer}'. "
            f"Valid options: {', '.join(sorted(_VALID_OPTIMIZERS))}"
        )

    if maxiter < 1:
        raise ValueError("maxiter must be at least 1")

    if boundary_tol < 0:
        raise ValueError("boundary_tol must be non-negative")

    for name, value in [
        ("check_nobs_vs_rankZ", check_nobs_vs_rankZ),
        ("check_nobs_vs_nlev", check_nobs_vs_nlev),
        ("check_nlev_gtreq_5", check_nlev_gtreq_5),
        ("check_nlev_gtr_1", check_nlev_gtr_1),
        ("check_scaleX", check_scaleX),
    ]:
        if value not in _VALID_CHECK_ACTIONS:
            raise ValueError(
                f"{name} must be one of {_VALID_CHECK_ACTIONS}, got '{value}'"
            )

    if check_rankX not in _VALID_RANKX_ACTIONS:
        raise ValueError(
            f"check_rankX must be one of {_VALID_RANKX_ACTIONS}, got '{check_rankX}'"
        )


def _build_scipy_options(
    *,
    optimizer: str,
    maxiter: int,
    ftol: float,
    gtol: float,
    xtol: float,
    opt_ctrl: dict[str, Any],
) -> dict[str, Any]:
    options: dict[str, Any] = {"maxiter": maxiter}

    if optimizer in ("L-BFGS-B", "BFGS"):
        options["gtol"] = gtol

    if optimizer == "L-BFGS-B":
        options["ftol"] = ftol

    if optimizer in ("Nelder-Mead", "Powell"):
        options["xatol"] = xtol
        options["fatol"] = ftol

    if optimizer == "trust-constr":
        options["gtol"] = gtol
        options["xtol"] = xtol

    options.update(opt_ctrl)
    return options


@dataclass
class LmerControl:
    """Control parameters for linear mixed model fitting.

    This class provides fine-grained control over the optimization
    process when fitting linear mixed-effects models with `lmer()`.

    Parameters
    ----------
    optimizer : str, default "bobyqa"
        Optimization algorithm to use. Options:
        - "bobyqa": BOBYQA (Bound Optimization BY Quadratic Approximation) -
          default, fastest and most reliable
        - "L-BFGS-B": Limited-memory BFGS with bounds
        - "BFGS": BFGS without bounds
        - "Nelder-Mead": Simplex algorithm
        - "Powell": Powell's method
        - "trust-constr": Trust-region constrained
    maxiter : int, default 1000
        Maximum number of iterations for the optimizer.
    ftol : float, default 1e-8
        Function tolerance for convergence.
    gtol : float, default 1e-5
        Gradient tolerance for convergence.
    xtol : float, default 1e-8
        Parameter tolerance for convergence.
    boundary_tol : float, default 1e-4
        Tolerance for detecting boundary (singular) fits.
        If any variance component is smaller than this value,
        the fit is considered singular.
    check_conv : bool, default True
        Whether to check convergence and warn if not converged.
    check_singular : bool, default True
        Whether to check for and warn about singular fits.
    calc_derivs : bool, default True
        Whether to calculate gradient/Hessian for variance-covariance.
    use_rust : bool, default True
        Whether to use Rust backend for optimization (if available).
    em_init : bool, default False
        Whether to use EM-REML algorithm for initialization before
        switching to direct optimization. Can improve convergence for
        difficult models. Supports random intercepts, random slopes
        (correlated and uncorrelated), and multiple random effect terms.
    em_maxiter : int, default 50
        Maximum EM iterations if em_init=True.
    optCtrl : dict, optional
        Additional options passed directly to the optimizer.
        For BOBYQA, supports: rhobeg, rhoend, seek_global_minimum.

    Examples
    --------
    >>> ctrl = LmerControl(optimizer="Nelder-Mead", maxiter=2000)
    >>> result = lmer("y ~ x + (1|group)", data, control=ctrl)

    >>> ctrl = LmerControl(optimizer="bobyqa")
    >>> result = lmer("y ~ x + (1|group)", data, control=ctrl)

    >>> ctrl = LmerControl(boundary_tol=1e-5, check_singular=False)
    >>> result = lmer("y ~ x + (x|group)", data, control=ctrl)
    """

    optimizer: str = "bobyqa"
    maxiter: int = 1000
    ftol: float = 1e-8
    gtol: float = 1e-5
    xtol: float = 1e-8
    boundary_tol: float = 1e-4
    check_conv: bool = True
    check_singular: bool = True
    calc_derivs: bool = True
    use_rust: bool = True
    em_init: bool = False
    em_maxiter: int = 50
    check_nobs_vs_rankZ: str = "ignore"
    check_nobs_vs_nlev: str = "ignore"
    check_nlev_gtreq_5: str = "warning"
    check_nlev_gtr_1: str = "stop"
    check_rankX: str = "message+drop.cols"
    check_scaleX: str = "warning"
    optCtrl: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_common_control(
            optimizer=self.optimizer,
            maxiter=self.maxiter,
            boundary_tol=self.boundary_tol,
            check_nobs_vs_rankZ=self.check_nobs_vs_rankZ,
            check_nobs_vs_nlev=self.check_nobs_vs_nlev,
            check_nlev_gtreq_5=self.check_nlev_gtreq_5,
            check_nlev_gtr_1=self.check_nlev_gtr_1,
            check_scaleX=self.check_scaleX,
            check_rankX=self.check_rankX,
        )

    def get_scipy_options(self) -> dict[str, Any]:
        """Get options dict for scipy.optimize.minimize."""
        return _build_scipy_options(
            optimizer=self.optimizer,
            maxiter=self.maxiter,
            ftol=self.ftol,
            gtol=self.gtol,
            xtol=self.xtol,
            opt_ctrl=self.optCtrl,
        )

    def __repr__(self) -> str:
        return (
            f"LmerControl(optimizer='{self.optimizer}', maxiter={self.maxiter}, "
            f"boundary_tol={self.boundary_tol})"
        )


@dataclass
class GlmerControl:
    """Control parameters for generalized linear mixed model fitting.

    This class provides fine-grained control over the optimization
    process when fitting generalized linear mixed-effects models with `glmer()`.

    Parameters
    ----------
    optimizer : str, default "bobyqa"
        Optimization algorithm to use. Options:
        - "bobyqa": BOBYQA (Bound Optimization BY Quadratic Approximation) -
          default, fastest and most reliable
        - "L-BFGS-B": Limited-memory BFGS with bounds
        - "BFGS": BFGS without bounds
        - "Nelder-Mead": Simplex algorithm
        - "Powell": Powell's method
        - "trust-constr": Trust-region constrained
    maxiter : int, default 1000
        Maximum number of iterations for the optimizer.
    ftol : float, default 1e-8
        Function tolerance for convergence.
    gtol : float, default 1e-5
        Gradient tolerance for convergence.
    xtol : float, default 1e-8
        Parameter tolerance for convergence.
    boundary_tol : float, default 1e-4
        Tolerance for detecting boundary (singular) fits.
    check_conv : bool, default True
        Whether to check convergence and warn if not converged.
    check_singular : bool, default True
        Whether to check for and warn about singular fits.
    tolPwrss : float, default 1e-7
        Tolerance for penalized weighted residual sum of squares
        convergence in the PIRLS algorithm.
    compDev : bool, default True
        Whether to compute deviance (vs just optimize parameters).
    nAGQ0initStep : bool, default True
        Whether to start with nAGQ=0 step before switching to
        the requested nAGQ value.
    em_init : bool, default False
        Whether to use EM-REML algorithm for initialization before
        switching to direct optimization. Runs a linear mixed model
        EM-REML to obtain starting theta values. Supports random intercepts,
        random slopes (correlated and uncorrelated), and multiple random
        effect terms.
    em_maxiter : int, default 50
        Maximum EM iterations if em_init=True.
    optCtrl : dict, optional
        Additional options passed directly to the optimizer.
        For BOBYQA, supports: rhobeg, rhoend, seek_global_minimum.

    Examples
    --------
    >>> ctrl = GlmerControl(optimizer="Nelder-Mead", maxiter=2000)
    >>> result = glmer("y ~ x + (1|group)", data, family=Binomial(), control=ctrl)

    >>> ctrl = GlmerControl(optimizer="bobyqa")
    >>> result = glmer("y ~ x + (1|group)", data, family=Binomial(), control=ctrl)

    >>> ctrl = GlmerControl(tolPwrss=1e-8, nAGQ0initStep=False)
    >>> result = glmer("y ~ x + (1|group)", data, family=Binomial(), control=ctrl)
    """

    optimizer: str = "bobyqa"
    maxiter: int = 1000
    ftol: float = 1e-8
    gtol: float = 1e-5
    xtol: float = 1e-8
    boundary_tol: float = 1e-4
    check_conv: bool = True
    check_singular: bool = True
    tolPwrss: float = 1e-7
    compDev: bool = True
    nAGQ0initStep: bool = True
    em_init: bool = False
    em_maxiter: int = 50
    check_nobs_vs_rankZ: str = "ignore"
    check_nobs_vs_nlev: str = "ignore"
    check_nlev_gtreq_5: str = "warning"
    check_nlev_gtr_1: str = "stop"
    check_rankX: str = "message+drop.cols"
    check_scaleX: str = "warning"
    optCtrl: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tolPwrss <= 0:
            raise ValueError("tolPwrss must be positive")
        _validate_common_control(
            optimizer=self.optimizer,
            maxiter=self.maxiter,
            boundary_tol=self.boundary_tol,
            check_nobs_vs_rankZ=self.check_nobs_vs_rankZ,
            check_nobs_vs_nlev=self.check_nobs_vs_nlev,
            check_nlev_gtreq_5=self.check_nlev_gtreq_5,
            check_nlev_gtr_1=self.check_nlev_gtr_1,
            check_scaleX=self.check_scaleX,
            check_rankX=self.check_rankX,
        )

    def get_scipy_options(self) -> dict[str, Any]:
        """Get options dict for scipy.optimize.minimize."""
        return _build_scipy_options(
            optimizer=self.optimizer,
            maxiter=self.maxiter,
            ftol=self.ftol,
            gtol=self.gtol,
            xtol=self.xtol,
            opt_ctrl=self.optCtrl,
        )

    def __repr__(self) -> str:
        return (
            f"GlmerControl(optimizer='{self.optimizer}', maxiter={self.maxiter}, "
            f"boundary_tol={self.boundary_tol}, tolPwrss={self.tolPwrss})"
        )


def lmerControl(
    optimizer: str = "bobyqa",
    maxiter: int = 1000,
    ftol: float = 1e-8,
    gtol: float = 1e-5,
    xtol: float = 1e-8,
    boundary_tol: float = 1e-4,
    check_conv: bool = True,
    check_singular: bool = True,
    calc_derivs: bool = True,
    use_rust: bool = True,
    em_init: bool = False,
    em_maxiter: int = 50,
    check_nobs_vs_rankZ: str = "ignore",
    check_nobs_vs_nlev: str = "ignore",
    check_nlev_gtreq_5: str = "warning",
    check_nlev_gtr_1: str = "stop",
    check_rankX: str = "message+drop.cols",
    check_scaleX: str = "warning",
    optCtrl: dict[str, Any] | None = None,
) -> LmerControl:
    """Create a control object for lmer().

    This is a convenience function that creates an LmerControl object.
    See LmerControl for parameter documentation.

    Returns
    -------
    LmerControl
        Control object for lmer().

    Examples
    --------
    >>> ctrl = lmerControl(optimizer="Nelder-Mead", maxiter=2000)
    >>> result = lmer("y ~ x + (1|group)", data, control=ctrl)
    """
    return LmerControl(
        optimizer=optimizer,
        maxiter=maxiter,
        ftol=ftol,
        gtol=gtol,
        xtol=xtol,
        boundary_tol=boundary_tol,
        check_conv=check_conv,
        check_singular=check_singular,
        calc_derivs=calc_derivs,
        use_rust=use_rust,
        em_init=em_init,
        em_maxiter=em_maxiter,
        check_nobs_vs_rankZ=check_nobs_vs_rankZ,
        check_nobs_vs_nlev=check_nobs_vs_nlev,
        check_nlev_gtreq_5=check_nlev_gtreq_5,
        check_nlev_gtr_1=check_nlev_gtr_1,
        check_rankX=check_rankX,
        check_scaleX=check_scaleX,
        optCtrl=optCtrl or {},
    )


def glmerControl(
    optimizer: str = "bobyqa",
    maxiter: int = 1000,
    ftol: float = 1e-8,
    gtol: float = 1e-5,
    xtol: float = 1e-8,
    boundary_tol: float = 1e-4,
    check_conv: bool = True,
    check_singular: bool = True,
    tolPwrss: float = 1e-7,
    compDev: bool = True,
    nAGQ0initStep: bool = True,
    em_init: bool = False,
    em_maxiter: int = 50,
    check_nobs_vs_rankZ: str = "ignore",
    check_nobs_vs_nlev: str = "ignore",
    check_nlev_gtreq_5: str = "warning",
    check_nlev_gtr_1: str = "stop",
    check_rankX: str = "message+drop.cols",
    check_scaleX: str = "warning",
    optCtrl: dict[str, Any] | None = None,
) -> GlmerControl:
    """Create a control object for glmer().

    This is a convenience function that creates a GlmerControl object.
    See GlmerControl for parameter documentation.

    Returns
    -------
    GlmerControl
        Control object for glmer().

    Examples
    --------
    >>> ctrl = glmerControl(optimizer="Nelder-Mead", maxiter=2000)
    >>> result = glmer("y ~ x + (1|group)", data, family=Binomial(), control=ctrl)
    """
    return GlmerControl(
        optimizer=optimizer,
        maxiter=maxiter,
        ftol=ftol,
        gtol=gtol,
        xtol=xtol,
        boundary_tol=boundary_tol,
        check_conv=check_conv,
        check_singular=check_singular,
        tolPwrss=tolPwrss,
        compDev=compDev,
        nAGQ0initStep=nAGQ0initStep,
        em_init=em_init,
        em_maxiter=em_maxiter,
        check_nobs_vs_rankZ=check_nobs_vs_rankZ,
        check_nobs_vs_nlev=check_nobs_vs_nlev,
        check_nlev_gtreq_5=check_nlev_gtreq_5,
        check_nlev_gtr_1=check_nlev_gtr_1,
        check_rankX=check_rankX,
        check_scaleX=check_scaleX,
        optCtrl=optCtrl or {},
    )
