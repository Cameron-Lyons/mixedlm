from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LmerControl:
    """Control parameters for linear mixed model fitting.

    This class provides fine-grained control over the optimization
    process when fitting linear mixed-effects models with `lmer()`.

    Parameters
    ----------
    optimizer : str, default "L-BFGS-B"
        Optimization algorithm to use. Options:
        - "L-BFGS-B": Limited-memory BFGS with bounds (default)
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
    optCtrl : dict, optional
        Additional options passed directly to scipy.optimize.minimize.

    Examples
    --------
    >>> ctrl = LmerControl(optimizer="Nelder-Mead", maxiter=2000)
    >>> result = lmer("y ~ x + (1|group)", data, control=ctrl)

    >>> ctrl = LmerControl(boundary_tol=1e-5, check_singular=False)
    >>> result = lmer("y ~ x + (x|group)", data, control=ctrl)
    """

    optimizer: str = "L-BFGS-B"
    maxiter: int = 1000
    ftol: float = 1e-8
    gtol: float = 1e-5
    xtol: float = 1e-8
    boundary_tol: float = 1e-4
    check_conv: bool = True
    check_singular: bool = True
    calc_derivs: bool = True
    use_rust: bool = True
    optCtrl: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        valid_optimizers = {
            "L-BFGS-B",
            "BFGS",
            "Nelder-Mead",
            "Powell",
            "trust-constr",
            "SLSQP",
            "TNC",
            "COBYLA",
        }
        if self.optimizer not in valid_optimizers:
            raise ValueError(
                f"Unknown optimizer '{self.optimizer}'. "
                f"Valid options: {', '.join(sorted(valid_optimizers))}"
            )

        if self.maxiter < 1:
            raise ValueError("maxiter must be at least 1")

        if self.boundary_tol < 0:
            raise ValueError("boundary_tol must be non-negative")

    def get_scipy_options(self) -> dict[str, Any]:
        """Get options dict for scipy.optimize.minimize."""
        options: dict[str, Any] = {"maxiter": self.maxiter}

        if self.optimizer in ("L-BFGS-B", "BFGS"):
            options["gtol"] = self.gtol

        if self.optimizer == "L-BFGS-B":
            options["ftol"] = self.ftol

        if self.optimizer in ("Nelder-Mead", "Powell"):
            options["xatol"] = self.xtol
            options["fatol"] = self.ftol

        if self.optimizer == "trust-constr":
            options["gtol"] = self.gtol
            options["xtol"] = self.xtol

        options.update(self.optCtrl)
        return options

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
    optimizer : str, default "L-BFGS-B"
        Optimization algorithm to use. Options:
        - "L-BFGS-B": Limited-memory BFGS with bounds (default)
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
    optCtrl : dict, optional
        Additional options passed directly to scipy.optimize.minimize.

    Examples
    --------
    >>> ctrl = GlmerControl(optimizer="Nelder-Mead", maxiter=2000)
    >>> result = glmer("y ~ x + (1|group)", data, family=Binomial(), control=ctrl)

    >>> ctrl = GlmerControl(tolPwrss=1e-8, nAGQ0initStep=False)
    >>> result = glmer("y ~ x + (1|group)", data, family=Binomial(), control=ctrl)
    """

    optimizer: str = "L-BFGS-B"
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
    optCtrl: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        valid_optimizers = {
            "L-BFGS-B",
            "BFGS",
            "Nelder-Mead",
            "Powell",
            "trust-constr",
            "SLSQP",
            "TNC",
            "COBYLA",
        }
        if self.optimizer not in valid_optimizers:
            raise ValueError(
                f"Unknown optimizer '{self.optimizer}'. "
                f"Valid options: {', '.join(sorted(valid_optimizers))}"
            )

        if self.maxiter < 1:
            raise ValueError("maxiter must be at least 1")

        if self.boundary_tol < 0:
            raise ValueError("boundary_tol must be non-negative")

        if self.tolPwrss <= 0:
            raise ValueError("tolPwrss must be positive")

    def get_scipy_options(self) -> dict[str, Any]:
        """Get options dict for scipy.optimize.minimize."""
        options: dict[str, Any] = {"maxiter": self.maxiter}

        if self.optimizer in ("L-BFGS-B", "BFGS"):
            options["gtol"] = self.gtol

        if self.optimizer == "L-BFGS-B":
            options["ftol"] = self.ftol

        if self.optimizer in ("Nelder-Mead", "Powell"):
            options["xatol"] = self.xtol
            options["fatol"] = self.ftol

        if self.optimizer == "trust-constr":
            options["gtol"] = self.gtol
            options["xtol"] = self.xtol

        options.update(self.optCtrl)
        return options

    def __repr__(self) -> str:
        return (
            f"GlmerControl(optimizer='{self.optimizer}', maxiter={self.maxiter}, "
            f"boundary_tol={self.boundary_tol}, tolPwrss={self.tolPwrss})"
        )


def lmerControl(
    optimizer: str = "L-BFGS-B",
    maxiter: int = 1000,
    ftol: float = 1e-8,
    gtol: float = 1e-5,
    xtol: float = 1e-8,
    boundary_tol: float = 1e-4,
    check_conv: bool = True,
    check_singular: bool = True,
    calc_derivs: bool = True,
    use_rust: bool = True,
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
        optCtrl=optCtrl or {},
    )


def glmerControl(
    optimizer: str = "L-BFGS-B",
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
        optCtrl=optCtrl or {},
    )
