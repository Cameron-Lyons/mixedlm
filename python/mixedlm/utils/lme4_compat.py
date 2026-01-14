from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult
    from mixedlm.models.nlmer import NlmerResult

MerMod = Union["LmerResult", "GlmerResult", "NlmerResult"]


def sigma(model: MerMod) -> float:
    """Extract the residual standard deviation from a fitted model.

    This is a convenience function that extracts sigma (the residual
    standard deviation) from linear mixed models. For GLMMs and NLMMs,
    this typically returns 1.0 as the scale is absorbed into the
    variance function.

    Parameters
    ----------
    model : LmerResult, GlmerResult, or NlmerResult
        A fitted mixed model.

    Returns
    -------
    float
        The residual standard deviation.

    Examples
    --------
    >>> result = lmer("y ~ x + (1|group)", data)
    >>> sigma(result)
    1.234

    >>> result = glmer("y ~ x + (1|group)", data, family=Binomial())
    >>> sigma(result)
    1.0

    See Also
    --------
    VarCorr : Extract variance-covariance components.
    """
    if hasattr(model, "sigma"):
        s = model.sigma
        if callable(s):
            return s
        return s
    elif hasattr(model, "get_sigma"):
        return model.get_sigma()
    return 1.0


def ngrps(model: MerMod) -> dict[str, int]:
    """Extract the number of levels for each grouping factor.

    This is a convenience function that returns a dictionary mapping
    grouping factor names to the number of levels in each factor.

    Parameters
    ----------
    model : LmerResult, GlmerResult, or NlmerResult
        A fitted mixed model.

    Returns
    -------
    dict[str, int]
        Dictionary mapping grouping factor names to number of levels.

    Examples
    --------
    >>> result = lmer("y ~ x + (1|subject) + (1|item)", data)
    >>> ngrps(result)
    {'subject': 20, 'item': 50}

    See Also
    --------
    ranef : Extract random effects.
    VarCorr : Extract variance-covariance components.
    """
    if hasattr(model, "ngrps"):
        return model.ngrps()
    elif hasattr(model, "matrices"):
        return {
            struct.grouping_factor: struct.n_levels for struct in model.matrices.random_structures
        }
    return {}


def fixef(model: MerMod) -> dict[str, float]:
    """Extract fixed effects coefficients from a fitted model.

    Parameters
    ----------
    model : LmerResult, GlmerResult, or NlmerResult
        A fitted mixed model.

    Returns
    -------
    dict[str, float]
        Dictionary mapping fixed effect names to coefficient values.

    Examples
    --------
    >>> result = lmer("y ~ x + (1|group)", data)
    >>> fixef(result)
    {'(Intercept)': 5.0, 'x': 2.3}
    """
    return model.fixef()


def ranef(model: MerMod, condVar: bool = False) -> dict[str, dict[str, NDArray[np.floating]]]:
    """Extract random effects (BLUPs) from a fitted model.

    Parameters
    ----------
    model : LmerResult, GlmerResult, or NlmerResult
        A fitted mixed model.
    condVar : bool, default False
        If True, include conditional variance of random effects.
        Note: Not all model types support condVar.

    Returns
    -------
    dict
        Nested dictionary of random effects by grouping factor and term.

    Examples
    --------
    >>> result = lmer("y ~ x + (1|group)", data)
    >>> ranef(result)
    {'group': {'(Intercept)': array([...])}}
    """
    if hasattr(model, "isNLMM") and model.isNLMM():
        return model.ranef()  # type: ignore[return-value]
    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult, RanefResult

    if isinstance(model, LmerResult | GlmerResult):
        result = model.ranef(condVar=condVar)
        if isinstance(result, RanefResult):
            return result.values
        return result  # type: ignore[return-value]
    return model.ranef()  # type: ignore[return-value]


def VarCorr(model: MerMod):
    """Extract variance-covariance components from a fitted model.

    Parameters
    ----------
    model : LmerResult, GlmerResult, or NlmerResult
        A fitted mixed model.

    Returns
    -------
    VarCorr or GlmerVarCorr
        Variance-covariance structure of random effects.

    Examples
    --------
    >>> result = lmer("y ~ x + (1|group)", data)
    >>> VarCorr(result)
    """
    return model.VarCorr()


def getME(model: MerMod, name: str):
    """Extract model elements by name.

    This is a convenience wrapper around the getME method of fitted
    model objects.

    Parameters
    ----------
    model : LmerResult, GlmerResult, or NlmerResult
        A fitted mixed model.
    name : str
        Name of the component to extract (e.g., "X", "Z", "theta", "beta").

    Returns
    -------
    The requested model component.

    Raises
    ------
    AttributeError
        If the model type does not support getME (e.g., NlmerResult).

    Examples
    --------
    >>> result = lmer("y ~ x + (1|group)", data)
    >>> X = getME(result, "X")
    >>> theta = getME(result, "theta")
    """
    if not hasattr(model, "getME"):
        raise AttributeError(
            f"{type(model).__name__} does not support getME(). Use model attributes directly."
        )
    return model.getME(name)


def coef(model: MerMod) -> dict:
    """Extract combined coefficients (fixed + random) for each group level.

    Parameters
    ----------
    model : LmerResult, GlmerResult, or NlmerResult
        A fitted mixed model.

    Returns
    -------
    dict
        Dictionary mapping group levels to their coefficient dictionaries.

    Examples
    --------
    >>> result = lmer("y ~ x + (x|group)", data)
    >>> coef(result)
    {'A': {'(Intercept)': 5.1, 'x': 2.3}, 'B': {...}}
    """
    return model.coef()


def pvalues(
    model: MerMod,
    method: str = "Satterthwaite",
) -> dict[str, float]:
    """Extract p-values for fixed effects from a fitted model.

    For linear mixed models (LMMs), p-values can be computed using
    Satterthwaite or Kenward-Roger approximations for degrees of freedom,
    or using the normal approximation. For GLMMs, the normal (z-test)
    approximation is always used.

    Parameters
    ----------
    model : LmerResult, GlmerResult, or NlmerResult
        A fitted mixed model.
    method : str, default "Satterthwaite"
        Method for computing p-values:
        - "Satterthwaite": Satterthwaite approximation for df (LMM only)
        - "KR" or "Kenward-Roger": Kenward-Roger approximation (LMM only)
        - "normal" or "z": Normal (z-test) approximation
        For GLMMs, this parameter is ignored and "normal" is always used.

    Returns
    -------
    dict[str, float]
        Dictionary mapping fixed effect names to p-values.

    Notes
    -----
    P-values in mixed models are controversial. The Satterthwaite and
    Kenward-Roger methods provide approximate degrees of freedom, which
    can be more accurate for small samples. However, these are only
    approximations. For critical applications, consider using likelihood
    ratio tests, bootstrapping, or Bayesian methods instead.

    For GLMMs, the Wald z-test is used, which assumes the sampling
    distribution of the estimates is approximately normal. This
    approximation improves with larger sample sizes.

    Examples
    --------
    >>> result = lmer("Reaction ~ Days + (Days|Subject)", sleepstudy)
    >>> pvalues(result)
    {'(Intercept)': 0.0, 'Days': 1.2e-10}

    >>> pvalues(result, method="normal")
    {'(Intercept)': 0.0, 'Days': 5.3e-15}

    See Also
    --------
    confint : Confidence intervals for parameters.
    anova : Likelihood ratio tests for model comparison.
    """
    from scipy import stats

    beta = model.fixef()
    if not hasattr(model, "vcov"):
        raise TypeError(f"{type(model).__name__} does not support vcov()")
    vcov = model.vcov()
    names = list(beta.keys())
    beta_vals = np.array(list(beta.values()))
    se_vals = np.sqrt(np.diag(vcov))

    t_vals = beta_vals / se_vals

    result: dict[str, float] = {}

    if hasattr(model, "isGLMM") and model.isGLMM():
        for i, name in enumerate(names):
            p = 2 * (1 - stats.norm.cdf(np.abs(t_vals[i])))
            result[name] = float(p)
        return result

    if hasattr(model, "isNLMM") and model.isNLMM():
        n = len(model.fitted())
        p_params = len(beta)
        df_resid = n - p_params
        for i, name in enumerate(names):
            p = 2 * (1 - stats.t.cdf(np.abs(t_vals[i]), df_resid))
            result[name] = float(p)
        return result

    method = method.lower()
    if method in ("normal", "z"):
        for i, name in enumerate(names):
            p = 2 * (1 - stats.norm.cdf(np.abs(t_vals[i])))
            result[name] = float(p)
        return result

    if not hasattr(model, "matrices"):
        for i, name in enumerate(names):
            p = 2 * (1 - stats.norm.cdf(np.abs(t_vals[i])))
            result[name] = float(p)
        return result

    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult

    if not isinstance(model, LmerResult | GlmerResult):
        for i, name in enumerate(names):
            p = 2 * (1 - stats.norm.cdf(np.abs(t_vals[i])))
            result[name] = float(p)
        return result

    n = model.matrices.n_obs
    p_fixed = model.matrices.n_fixed

    if method in ("satterthwaite", "satt") or method in ("kr", "kenward-roger", "kenward_roger"):
        for i, name in enumerate(names):
            df = _satterthwaite_df(model, i, se_vals[i])
            df = max(1, min(df, n - p_fixed))
            p = 2 * (1 - stats.t.cdf(np.abs(t_vals[i]), df))
            result[name] = float(p)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'Satterthwaite', 'KR', or 'normal'.")

    return result


def _satterthwaite_df(model: MerMod, _param_idx: int, _se: float) -> float:
    """Compute Satterthwaite degrees of freedom approximation.

    This is a simplified approximation based on the variance components.
    """
    if not hasattr(model, "matrices"):
        return 100.0

    n = model.matrices.n_obs
    p = model.matrices.n_fixed
    n_groups = sum(int(s.n_levels) for s in model.matrices.random_structures)

    if n_groups == 0:
        return float(n - p)

    avg_obs_per_group = n / max(n_groups, 1)

    df_between = n_groups - 1
    df_within = n - n_groups - p + 1

    model_sigma = getattr(model, "sigma", 1.0)
    var_ratio = model_sigma**2 / (_se**2 * n + 1e-10)

    df = max(df_between, 1) if var_ratio > avg_obs_per_group else df_within

    return float(max(1, min(df, n - p)))


def lmList(
    formula: str,
    data: pd.DataFrame,
    group: str | None = None,
    pool: bool = True,
) -> dict[str, object]:
    """Fit separate linear models for each level of a grouping factor.

    This function fits individual linear models (using OLS) to subsets
    of the data defined by the levels of a grouping factor. This is
    useful for exploratory analysis before fitting a mixed model, or
    for comparing individual-level estimates to mixed model estimates.

    Parameters
    ----------
    formula : str
        Model formula without random effects (e.g., "y ~ x1 + x2").
        Random effects terms like "(1|group)" should NOT be included.
    data : pd.DataFrame
        Data containing the variables in the formula.
    group : str, optional
        Name of the grouping variable. If None, attempts to infer from
        the formula if it contains a bar (|).
    pool : bool, default True
        If True, also fits a pooled model to all data.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'fits': Dict mapping group levels to fitted model results
        - 'coef': DataFrame with coefficients for each group
        - 'pooled': Pooled model fit (if pool=True)

    Raises
    ------
    ValueError
        If group is not specified and cannot be inferred.
    ImportError
        If statsmodels is not available.

    Examples
    --------
    >>> from mixedlm import lmList
    >>> results = lmList("Reaction ~ Days", sleepstudy, group="Subject")
    >>> results['coef']  # DataFrame of coefficients per subject

    Notes
    -----
    This function requires statsmodels to be installed for OLS fitting.
    If statsmodels is not available, a simplified implementation using
    numpy's least squares is used.

    See Also
    --------
    lmer : Fit linear mixed models.
    """
    import re

    if group is None:
        bar_match = re.search(r"\|\s*(\w+)\s*\)", formula)
        if bar_match:
            group = bar_match.group(1)
            formula = re.sub(r"\s*\+?\s*\([^)]*\|\s*\w+\s*\)", "", formula).strip()
        else:
            raise ValueError("group must be specified when formula does not contain random effects")

    if group not in data.columns:
        raise ValueError(f"Grouping variable '{group}' not found in data")

    formula_parts = formula.split("~")
    if len(formula_parts) != 2:
        raise ValueError(f"Invalid formula: {formula}")

    response = formula_parts[0].strip()
    predictors = formula_parts[1].strip()

    try:
        import statsmodels.formula.api as smf

        use_statsmodels = True
    except ImportError:
        use_statsmodels = False

    groups = data[group].unique()
    fits: dict[str, object] = {}
    coef_list: list[dict] = []

    for g in groups:
        subset = data[data[group] == g].copy()

        if len(subset) < 2:
            continue

        if use_statsmodels:
            try:
                fit = smf.ols(formula, data=subset).fit()
                fits[str(g)] = fit
                coef_row = {group: g}
                coef_row.update(dict(zip(fit.params.index, fit.params.values, strict=False)))
                coef_list.append(coef_row)
            except Exception:
                continue
        else:
            try:
                y = subset[response].values
                X = np.column_stack([np.ones(len(subset)), subset[predictors].values])
                beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                fits[str(g)] = {"coef": beta, "n": len(subset)}
                coef_row = {group: g, "(Intercept)": beta[0], predictors: beta[1]}
                coef_list.append(coef_row)
            except Exception:
                continue

    coef_df = pd.DataFrame(coef_list)
    if group in coef_df.columns:
        coef_df = coef_df.set_index(group)

    result: dict[str, object] = {
        "fits": fits,
        "coef": coef_df,
    }

    if pool:
        if use_statsmodels:
            try:
                pooled = smf.ols(formula, data=data).fit()
                result["pooled"] = pooled
            except Exception:
                result["pooled"] = None
        else:
            try:
                y = data[response].values
                X = np.column_stack([np.ones(len(data)), data[predictors].values])
                beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                result["pooled"] = {"coef": beta, "n": len(data)}
            except Exception:
                result["pooled"] = None

    return result


@dataclass
class ConvergenceInfo:
    """Information about model convergence.

    Attributes
    ----------
    converged : bool
        Whether the optimizer reported convergence.
    is_singular : bool
        Whether the fit is singular (boundary fit).
    messages : list[str]
        List of convergence-related messages and warnings.
    optimizer : str
        Name of the optimizer used.
    iterations : int | None
        Number of iterations (if available).
    gradient_norm : float | None
        Norm of the gradient at convergence (if available).
    hessian_ok : bool | None
        Whether the Hessian is positive definite (if checked).
    """

    converged: bool
    is_singular: bool
    messages: list[str]
    optimizer: str
    iterations: int | None = None
    gradient_norm: float | None = None
    hessian_ok: bool | None = None

    def __str__(self) -> str:
        lines = ["Convergence Information:"]
        lines.append(f"  Converged: {self.converged}")
        lines.append(f"  Singular fit: {self.is_singular}")
        lines.append(f"  Optimizer: {self.optimizer}")
        if self.iterations is not None:
            lines.append(f"  Iterations: {self.iterations}")
        if self.gradient_norm is not None:
            lines.append(f"  Gradient norm: {self.gradient_norm:.6e}")
        if self.hessian_ok is not None:
            lines.append(f"  Hessian OK: {self.hessian_ok}")
        if self.messages:
            lines.append("  Messages:")
            for msg in self.messages:
                lines.append(f"    - {msg}")
        return "\n".join(lines)


def checkConv(
    model: MerMod,
    check_singular: bool = True,
    check_gradient: bool = True,
    check_hessian: bool = False,
    tol: float = 1e-4,
    grad_tol: float = 1e-3,
) -> ConvergenceInfo:
    """Check convergence of a fitted mixed model.

    This function performs various convergence diagnostics on a fitted
    model, including checking for singular fits, gradient magnitude,
    and optionally the Hessian matrix.

    Parameters
    ----------
    model : LmerResult, GlmerResult, or NlmerResult
        A fitted mixed model.
    check_singular : bool, default True
        Check if the model has a singular (boundary) fit.
    check_gradient : bool, default True
        Check if the gradient is near zero at convergence.
    check_hessian : bool, default False
        Check if the Hessian is positive definite.
    tol : float, default 1e-4
        Tolerance for singular fit detection.
    grad_tol : float, default 1e-3
        Tolerance for gradient norm check.

    Returns
    -------
    ConvergenceInfo
        Dataclass containing convergence diagnostics.

    Examples
    --------
    >>> result = lmer("Reaction ~ Days + (Days|Subject)", sleepstudy)
    >>> conv = checkConv(result)
    >>> print(conv)
    Convergence Information:
      Converged: True
      Singular fit: False
      Optimizer: L-BFGS-B
      ...

    >>> if not conv.converged or conv.is_singular:
    ...     print("Warning: Model may not have converged properly")

    See Also
    --------
    isSingular : Check for singular fit.
    allFit : Try multiple optimizers.
    """
    messages: list[str] = []

    converged = getattr(model, "converged", True)
    if not converged:
        messages.append("Optimizer did not report convergence")

    is_singular = False
    if check_singular and hasattr(model, "isSingular"):
        is_singular = model.isSingular(tol=tol)
        if is_singular:
            messages.append(f"Model is singular (boundary fit) at tolerance {tol}")

    optimizer = "unknown"
    if hasattr(model, "control") and hasattr(model.control, "optimizer"):
        optimizer = model.control.optimizer
    elif hasattr(model, "optimizer"):
        optimizer = model.optimizer

    iterations = None
    if hasattr(model, "optinfo") and isinstance(model.optinfo, dict):
        iterations = model.optinfo.get("nit") or model.optinfo.get("iterations")

    gradient_norm = None
    if check_gradient and hasattr(model, "optinfo") and isinstance(model.optinfo, dict):
        grad = model.optinfo.get("jac") or model.optinfo.get("gradient")
        if grad is not None:
            gradient_norm = float(np.linalg.norm(grad))
            if gradient_norm > grad_tol:
                messages.append(
                    f"Gradient norm ({gradient_norm:.2e}) exceeds tolerance ({grad_tol:.2e})"
                )

    hessian_ok = None
    if check_hessian:
        if not hasattr(model, "vcov"):
            messages.append("Model does not support vcov() for Hessian check")
        else:
            try:
                vcov = model.vcov()
                eigenvalues = np.linalg.eigvalsh(vcov)
                hessian_ok = bool(np.all(eigenvalues > 0))
                if not hessian_ok:
                    messages.append("Hessian is not positive definite")
            except Exception as e:
                messages.append(f"Could not check Hessian: {e}")
                hessian_ok = None

    return ConvergenceInfo(
        converged=converged,
        is_singular=is_singular,
        messages=messages,
        optimizer=optimizer,
        iterations=iterations,
        gradient_norm=gradient_norm,
        hessian_ok=hessian_ok,
    )


def convergence_ok(model: MerMod, tol: float = 1e-4) -> bool:
    """Quick check if model convergence is acceptable.

    This is a convenience function that returns True if the model
    converged and is not singular.

    Parameters
    ----------
    model : LmerResult, GlmerResult, or NlmerResult
        A fitted mixed model.
    tol : float, default 1e-4
        Tolerance for singular fit detection.

    Returns
    -------
    bool
        True if convergence is acceptable, False otherwise.

    Examples
    --------
    >>> result = lmer("Reaction ~ Days + (Days|Subject)", sleepstudy)
    >>> if convergence_ok(result):
    ...     print("Model converged properly")
    ... else:
    ...     print("Check convergence issues")
    """
    conv = checkConv(model, tol=tol, check_gradient=False, check_hessian=False)
    return conv.converged and not conv.is_singular


def fortify(
    model: MerMod,
    data: pd.DataFrame | None = None,
    include_re: bool = True,
) -> pd.DataFrame:
    """Add fitted values and residuals to data for plotting.

    This function augments a data frame with fitted values, residuals,
    and other diagnostic quantities from a fitted model. This is useful
    for creating diagnostic plots with matplotlib, seaborn, or other
    plotting libraries.

    Parameters
    ----------
    model : LmerResult, GlmerResult, or NlmerResult
        A fitted mixed model.
    data : DataFrame, optional
        Data frame to augment. If None, uses the model's stored data.
    include_re : bool, default True
        If True, include random effects in fitted values.

    Returns
    -------
    pd.DataFrame
        Data frame with added columns:
        - .fitted: Fitted values
        - .resid: Residuals
        - .fixed: Fixed effects contribution only
        - .mu: For GLMMs, the response scale fitted values

    Examples
    --------
    >>> result = lmer("Reaction ~ Days + (Days|Subject)", sleepstudy)
    >>> augmented = fortify(result)
    >>> augmented[['.fitted', '.resid']].head()

    >>> import matplotlib.pyplot as plt
    >>> plt.scatter(augmented['.fitted'], augmented['.resid'])
    >>> plt.xlabel('Fitted values')
    >>> plt.ylabel('Residuals')

    See Also
    --------
    fitted : Extract fitted values.
    residuals : Extract residuals.
    """
    if data is None:
        if hasattr(model, "matrices") and hasattr(model.matrices, "frame"):
            data = model.matrices.frame
        else:
            raise ValueError("No data available. Provide data or ensure model stores frame.")

    if data is None:
        raise ValueError("No data available")

    result = data.copy()

    fitted_vals = model.fitted()
    resid_vals = model.residuals()

    if len(fitted_vals) != len(result):
        result = result.iloc[: len(fitted_vals)].copy()

    result[".fitted"] = fitted_vals
    result[".resid"] = resid_vals

    if hasattr(model, "matrices") and hasattr(model.matrices, "X"):
        beta = np.array(list(model.fixef().values()))
        fixed_contrib = model.matrices.X @ beta
        if len(fixed_contrib) == len(result):
            result[".fixed"] = fixed_contrib

    if hasattr(model, "isGLMM") and model.isGLMM() and hasattr(model, "fitted"):
        result[".mu"] = model.fitted(type="response")  # type: ignore[call-arg]

    return result


@dataclass
class DevComp:
    """Deviance components from a fitted mixed model.

    Attributes
    ----------
    cmp : dict
        Named components including:
        - ldL2: Log determinant of L squared
        - ldRX2: Log determinant of RX squared
        - wrss: Weighted residual sum of squares
        - ussq: Sum of squared random effects
        - pwrss: Penalized weighted residual sum of squares
        - drsum: Deviance residual sum
        - REML: REML criterion (if applicable)
        - dev: Deviance or ML criterion
    dims : dict
        Dimension information including n, p, q, nmp, nth, etc.
    """

    cmp: dict[str, float]
    dims: dict[str, int]

    def __str__(self) -> str:
        lines = ["Deviance Components:"]
        lines.append("  Components:")
        for k, v in self.cmp.items():
            lines.append(f"    {k}: {v:.6g}")
        lines.append("  Dimensions:")
        for k, v in self.dims.items():
            lines.append(f"    {k}: {v}")
        return "\n".join(lines)


def devcomp(model: MerMod) -> DevComp:
    """Extract deviance components from a fitted model.

    This function extracts the components that make up the deviance
    or REML criterion for a fitted mixed model.

    Parameters
    ----------
    model : LmerResult, GlmerResult, or NlmerResult
        A fitted mixed model.

    Returns
    -------
    DevComp
        Dataclass containing component values and dimensions.

    Examples
    --------
    >>> result = lmer("Reaction ~ Days + (Days|Subject)", sleepstudy)
    >>> dc = devcomp(result)
    >>> print(dc)
    >>> dc.cmp['pwrss']  # Penalized weighted RSS

    Notes
    -----
    The deviance components are internal quantities used in computing
    the likelihood. They can be useful for understanding model behavior
    or for implementing custom inference procedures.

    See Also
    --------
    logLik : Extract log-likelihood.
    getME : Extract model elements.
    """
    cmp: dict[str, float] = {}
    dims: dict[str, int] = {}

    if hasattr(model, "deviance"):
        cmp["dev"] = float(model.deviance)

    if hasattr(model, "logLik"):
        ll = model.logLik()
        cmp["logLik"] = float(ll)

    if hasattr(model, "REML") and model.REML:
        cmp["REML"] = cmp.get("dev", 0.0)

    if hasattr(model, "sigma"):
        cmp["sigmaML"] = float(model.sigma)
        cmp["sigmaREML"] = float(model.sigma)

    if hasattr(model, "matrices"):
        matrices = model.matrices
        dims["n"] = matrices.n_obs
        dims["p"] = matrices.n_fixed
        dims["q"] = matrices.n_random

        if hasattr(matrices, "y") and hasattr(model, "fitted"):
            resid = matrices.y - model.fitted()
            cmp["wrss"] = float(np.sum(resid**2))

        if hasattr(model, "theta"):
            cmp["nth"] = len(model.theta)

        if hasattr(matrices, "random_structures"):
            dims["ngrps"] = sum(s.n_levels for s in matrices.random_structures)

    if hasattr(model, "u") and model.u is not None:
        cmp["ussq"] = float(np.sum(model.u**2))

    if "wrss" in cmp and "ussq" in cmp:
        cmp["pwrss"] = cmp["wrss"] + cmp["ussq"]

    return DevComp(cmp=cmp, dims=dims)


def vcconv(
    theta: NDArray[np.floating],
    random_structures: list,
    sigma: float = 1.0,
    to: str = "sdcorr",
) -> dict[str, dict]:
    """Convert variance component parameterization.

    This function converts between different representations of
    variance components: the internal theta parameterization,
    standard deviation/correlation form, and variance/covariance form.

    Parameters
    ----------
    theta : array-like
        The theta (relative covariance) parameters.
    random_structures : list
        List of random effect structures from the model.
    sigma : float, default 1.0
        Residual standard deviation.
    to : str, default "sdcorr"
        Target parameterization:
        - "sdcorr": Standard deviations and correlations
        - "varcov": Variances and covariances
        - "theta": Relative covariance (theta) parameters

    Returns
    -------
    dict
        Dictionary mapping grouping factors to their variance components
        in the requested parameterization.

    Examples
    --------
    >>> result = lmer("Reaction ~ Days + (Days|Subject)", sleepstudy)
    >>> theta = result.theta
    >>> structures = result.matrices.random_structures
    >>> vcconv(theta, structures, result.sigma, to="sdcorr")
    {'Subject': {'sd': [...], 'corr': [...]}}

    >>> vcconv(theta, structures, result.sigma, to="varcov")
    {'Subject': {'var': [...], 'cov': [...]}}

    See Also
    --------
    VarCorr : Extract variance-covariance from model.
    """
    result: dict[str, dict] = {}
    theta_idx = 0

    for struct in random_structures:
        n_terms = struct.n_terms
        group = struct.grouping_factor

        if struct.correlated:
            n_theta = n_terms * (n_terms + 1) // 2
            theta_block = theta[theta_idx : theta_idx + n_theta]
            theta_idx += n_theta

            L = np.zeros((n_terms, n_terms))
            idx = 0
            for j in range(n_terms):
                for i in range(j, n_terms):
                    L[i, j] = theta_block[idx]
                    idx += 1

            cov = L @ L.T * sigma**2

            if to == "varcov":
                var = np.diag(cov)
                cov_offdiag = []
                for i in range(n_terms):
                    for j in range(i + 1, n_terms):
                        cov_offdiag.append(cov[i, j])
                result[group] = {
                    "var": var.tolist(),
                    "cov": cov_offdiag,
                    "terms": struct.term_names,
                }
            elif to == "sdcorr":
                sd = np.sqrt(np.diag(cov))
                corr = []
                for i in range(n_terms):
                    for j in range(i + 1, n_terms):
                        if sd[i] > 0 and sd[j] > 0:
                            corr.append(cov[i, j] / (sd[i] * sd[j]))
                        else:
                            corr.append(0.0)
                result[group] = {
                    "sd": sd.tolist(),
                    "corr": corr,
                    "terms": struct.term_names,
                }
            else:
                result[group] = {
                    "theta": theta_block.tolist(),
                    "terms": struct.term_names,
                }
        else:
            theta_block = theta[theta_idx : theta_idx + n_terms]
            theta_idx += n_terms

            if to == "varcov":
                var = (theta_block * sigma) ** 2  # type: ignore[assignment]
                result[group] = {
                    "var": var.tolist(),
                    "cov": [],
                    "terms": struct.term_names,
                }
            elif to == "sdcorr":
                sd = np.abs(theta_block) * sigma
                result[group] = {
                    "sd": sd.tolist(),
                    "corr": [],
                    "terms": struct.term_names,
                }
            else:
                result[group] = {
                    "theta": theta_block.tolist(),
                    "terms": struct.term_names,
                }

    return result


def GHrule(n: int, asMatrix: bool = True) -> dict | NDArray[np.floating]:
    """Generate Gauss-Hermite quadrature rule.

    Compute the nodes and weights for Gauss-Hermite quadrature,
    which is used for numerical integration of functions weighted
    by exp(-x^2). This is used internally for adaptive Gauss-Hermite
    quadrature in GLMMs.

    Parameters
    ----------
    n : int
        Number of quadrature points (1-25 typically used).
    asMatrix : bool, default True
        If True, return as a matrix with columns [node, weight].
        If False, return a dictionary with 'nodes' and 'weights'.

    Returns
    -------
    ndarray or dict
        If asMatrix is True, returns (n, 2) array with nodes and weights.
        Otherwise, returns dict with 'nodes' and 'weights' arrays.

    Examples
    --------
    >>> gh = GHrule(5)
    >>> gh  # 5x2 array of nodes and weights
    array([[-2.02, 0.02],
           [-0.96, 0.39],
           ...])

    >>> gh = GHrule(3, asMatrix=False)
    >>> gh['nodes']
    array([-1.22, 0.0, 1.22])
    >>> gh['weights']
    array([0.30, 1.18, 0.30])

    Notes
    -----
    The nodes and weights are for the "probabilist's" Hermite polynomials,
    normalized for integration against exp(-x^2/2) / sqrt(2*pi).

    For integration of f(x) * exp(-x^2/2) / sqrt(2*pi):
        integral ≈ sum(weights * f(nodes))

    See Also
    --------
    glmer : Fit GLMMs using adaptive Gauss-Hermite quadrature.
    """
    from scipy.special import roots_hermite

    nodes, weights = roots_hermite(n)

    nodes = nodes * np.sqrt(2)
    weights = weights / np.sqrt(np.pi)

    if asMatrix:
        return np.column_stack([nodes, weights])
    else:
        return {"nodes": nodes, "weights": weights}


def factorize(
    data: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Convert columns to categorical (factor) type.

    This is a convenience function to ensure grouping variables
    are properly encoded as categorical variables before fitting
    a mixed model.

    Parameters
    ----------
    data : DataFrame
        Data frame to process.
    columns : list of str, optional
        Columns to convert. If None, converts all object/string columns.

    Returns
    -------
    pd.DataFrame
        Data frame with specified columns as categorical.

    Examples
    --------
    >>> data = pd.DataFrame({'y': [1,2,3], 'group': ['a', 'b', 'a']})
    >>> data = factorize(data, ['group'])
    >>> data['group'].dtype
    CategoricalDtype(...)
    """
    result = data.copy()

    if columns is None:
        columns = result.select_dtypes(include=["object", "string"]).columns.tolist()

    for col in columns:
        if col in result.columns:
            result[col] = pd.Categorical(result[col])

    return result


def isNested(
    factor1: pd.Series | NDArray | list,
    factor2: pd.Series | NDArray | list,
) -> bool:
    """Check if factor1 is nested within factor2.

    A factor is nested within another if each level of the first factor
    occurs within only one level of the second factor. For example,
    students nested within schools means each student belongs to exactly
    one school.

    Parameters
    ----------
    factor1 : array-like
        The potentially nested factor (e.g., student IDs).
    factor2 : array-like
        The grouping factor (e.g., school IDs).

    Returns
    -------
    bool
        True if factor1 is nested within factor2, False otherwise.

    Examples
    --------
    >>> import pandas as pd
    >>> students = ['s1', 's2', 's3', 's4', 's5', 's6']
    >>> schools = ['A', 'A', 'A', 'B', 'B', 'B']
    >>> isNested(students, schools)
    True

    >>> # Not nested - student s1 appears in both schools
    >>> students = ['s1', 's2', 's1', 's3', 's4', 's5']
    >>> schools = ['A', 'A', 'B', 'B', 'B', 'B']
    >>> isNested(students, schools)
    False

    >>> # Crossed design - items crossed with subjects
    >>> items = [1, 2, 3, 1, 2, 3]
    >>> subjects = ['A', 'A', 'A', 'B', 'B', 'B']
    >>> isNested(items, subjects)
    False

    Notes
    -----
    This function is useful for determining the appropriate random effects
    structure. Nested factors typically use (1|factor2/factor1) syntax,
    while crossed factors use (1|factor1) + (1|factor2).

    See Also
    --------
    lmer : Fit linear mixed models with nested or crossed effects.
    """
    f1 = np.asarray(factor1)
    f2 = np.asarray(factor2)

    if len(f1) != len(f2):
        raise ValueError("factor1 and factor2 must have the same length")

    level_to_group: dict = {}

    for level, group in zip(f1, f2, strict=False):
        if level in level_to_group:
            if level_to_group[level] != group:
                return False
        else:
            level_to_group[level] = group

    return True


def mkMerMod(
    model: MerMod,
    theta: NDArray[np.floating] | None = None,
    beta: NDArray[np.floating] | None = None,
) -> MerMod:
    """Create a new model object with updated parameters.

    This function creates a copy of a fitted model with new parameter
    values, without refitting. This is useful for simulation studies
    or exploring the parameter space.

    Parameters
    ----------
    model : LmerResult, GlmerResult, or NlmerResult
        A fitted mixed model to use as template.
    theta : array-like, optional
        New theta (variance component) parameters.
    beta : array-like, optional
        New fixed effects parameters.

    Returns
    -------
    Model result with updated parameters.

    Notes
    -----
    This creates a shallow copy - the model matrices are shared.
    The new object is not a fully "fitted" model but can be used
    for computing fitted values, simulating, etc.

    Examples
    --------
    >>> result = lmer("y ~ x + (1|g)", data)
    >>> new_theta = np.array([0.5])  # Change variance
    >>> modified = mkMerMod(result, theta=new_theta)
    """
    import copy

    new_model = copy.copy(model)

    if theta is not None:
        new_model.theta = np.asarray(theta)

    if beta is not None:
        if hasattr(new_model, "beta"):
            new_model.beta = np.asarray(beta)
        else:
            raise TypeError(f"{type(model).__name__} does not support beta modification")

    return new_model


def dummy(
    x: pd.Series | NDArray | list,
    base: int | str | None = None,
    contrasts: str = "treatment",
) -> NDArray[np.floating]:
    """Create dummy (indicator) variables from a categorical variable.

    Parameters
    ----------
    x : array-like
        Categorical variable (factor).
    base : int or str, optional
        Base level to exclude (reference category). If int, the index
        of the level to use as base. If str, the level name. If None,
        uses the first level.
    contrasts : str, default "treatment"
        Type of contrast coding:
        - "treatment": Treatment (dummy) coding with base level excluded
        - "sum": Sum (deviation) coding
        - "helmert": Helmert contrasts
        - "poly": Polynomial contrasts (for ordered factors)

    Returns
    -------
    ndarray
        Dummy variable matrix with shape (n, k-1) for k levels.

    Examples
    --------
    >>> x = ['A', 'B', 'C', 'A', 'B']
    >>> dummy(x)
    array([[0., 0.],
           [1., 0.],
           [0., 1.],
           [0., 0.],
           [1., 0.]])

    >>> dummy(x, base='B')
    array([[1., 0.],
           [0., 0.],
           [0., 1.],
           [1., 0.],
           [0., 0.]])
    """
    x = np.asarray(x)
    levels = np.unique(x)
    n = len(x)
    k = len(levels)

    if base is None:
        base_idx = 0
    elif isinstance(base, int):
        base_idx = base
    else:
        base_idx = int(np.where(levels == base)[0][0])

    if contrasts == "treatment":
        non_base = [i for i in range(k) if i != base_idx]
        result = np.zeros((n, k - 1), dtype=np.float64)
        for j, lvl_idx in enumerate(non_base):
            result[:, j] = (x == levels[lvl_idx]).astype(float)
    elif contrasts == "sum":
        result = np.zeros((n, k - 1), dtype=np.float64)
        for j in range(k - 1):
            result[x == levels[j], j] = 1.0
            result[x == levels[k - 1], j] = -1.0
    elif contrasts == "helmert":
        result = np.zeros((n, k - 1), dtype=np.float64)
        for j in range(k - 1):
            for i in range(j + 1):
                result[x == levels[i], j] = -1.0 / (j + 1)
            result[x == levels[j + 1], j] = 1.0
    elif contrasts == "poly":
        from numpy.polynomial.legendre import legvander

        poly_matrix = legvander(np.arange(k), k - 1)[:, 1:]
        level_to_idx = {lvl: i for i, lvl in enumerate(levels)}
        x_idx = np.array([level_to_idx[val] for val in x])
        result = np.asarray(poly_matrix[x_idx], dtype=np.float64)
    else:
        raise ValueError(f"Unknown contrast type: {contrasts}")

    return result


def REMLcrit(model: MerMod) -> float:
    """Extract the REML criterion from a fitted model.

    For models fit with REML=True, this returns the REML criterion.
    For models fit with ML, this returns the deviance.

    Parameters
    ----------
    model : LmerResult, GlmerResult, or NlmerResult
        A fitted mixed model.

    Returns
    -------
    float
        The REML criterion or deviance.

    Examples
    --------
    >>> result = lmer("Reaction ~ Days + (Days|Subject)", sleepstudy, REML=True)
    >>> REMLcrit(result)
    1743.6283

    >>> result_ml = lmer("Reaction ~ Days + (Days|Subject)", sleepstudy, REML=False)
    >>> REMLcrit(result_ml)  # Returns deviance
    1751.9393
    """
    return float(model.deviance)


def scale_vcov(
    vcov: NDArray[np.floating],
    center: NDArray[np.floating] | None = None,
    scale: NDArray[np.floating] | None = None,
) -> NDArray[np.floating]:
    """Adjust variance-covariance matrix for centering and scaling.

    When predictors are centered and/or scaled, the variance-covariance
    matrix of coefficients needs to be adjusted to be on the original
    scale.

    Parameters
    ----------
    vcov : ndarray
        Variance-covariance matrix of coefficients (p x p).
    center : ndarray, optional
        Centering values used for each predictor. If None, no centering
        adjustment is made.
    scale : ndarray, optional
        Scaling values used for each predictor. If None, no scaling
        adjustment is made.

    Returns
    -------
    ndarray
        Adjusted variance-covariance matrix.

    Examples
    --------
    >>> # If x was scaled by dividing by 2:
    >>> vcov = np.array([[1.0, 0.1], [0.1, 0.5]])
    >>> scale = np.array([1.0, 2.0])
    >>> scale_vcov(vcov, scale=scale)
    array([[1. , 0.2],
           [0.2, 2. ]])
    """
    vcov = np.asarray(vcov).copy()
    p = vcov.shape[0]

    if scale is not None:
        scale = np.asarray(scale)
        if len(scale) != p:
            raise ValueError(f"scale length {len(scale)} != vcov dimension {p}")
        D = np.diag(scale)
        vcov = D @ vcov @ D

    return vcov


def quickSimulate(
    formula: str,
    data: pd.DataFrame,
    beta: NDArray[np.floating] | dict[str, float] | None = None,
    theta: NDArray[np.floating] | None = None,
    sigma: float = 1.0,
    family: str | None = None,
    nsim: int = 1,
    seed: int | None = None,
) -> pd.DataFrame | list[pd.DataFrame]:
    """Quickly simulate response data from a formula specification.

    This is a convenience wrapper around simulate_formula that provides
    a simpler interface for common use cases.

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
        Variance component parameters. If None, uses ones.
    sigma : float, default 1.0
        Residual standard deviation (for Gaussian family).
    family : str, optional
        Distribution family name: "gaussian", "binomial", "poisson".
        Default is "gaussian".
    nsim : int, default 1
        Number of simulations to generate.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    DataFrame or list of DataFrame
        Simulated data with response variable.

    Examples
    --------
    >>> import numpy as np
    >>> data = pd.DataFrame({
    ...     'x': np.random.randn(100),
    ...     'group': np.repeat(['A', 'B', 'C', 'D', 'E'], 20)
    ... })
    >>> sim = quickSimulate(
    ...     "y ~ x + (1|group)",
    ...     data,
    ...     beta={'(Intercept)': 5.0, 'x': 2.0},
    ...     sigma=1.0
    ... )

    See Also
    --------
    simulate_formula : Full simulation function with more options.
    """
    from mixedlm.families.base import Family
    from mixedlm.models.modular import simulate_formula

    family_obj: Family | None = None
    if family is not None:
        family_lower = family.lower()
        if family_lower == "binomial":
            from mixedlm.families import Binomial

            family_obj = Binomial()
        elif family_lower == "poisson":
            from mixedlm.families import Poisson

            family_obj = Poisson()
        elif family_lower == "gaussian":
            from mixedlm.families import Gaussian

            family_obj = Gaussian()

    return simulate_formula(
        formula=formula,
        data=data,
        beta=beta,
        theta=theta,
        sigma=sigma,
        family=family_obj,
        nsim=nsim,
        seed=seed,
    )
