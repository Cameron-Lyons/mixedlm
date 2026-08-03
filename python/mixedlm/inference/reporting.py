from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult
    from mixedlm.models.nlmer import NlmerResult

    MixedModelResult: TypeAlias = LmerResult | GlmerResult | NlmerResult
else:
    MixedModelResult: TypeAlias = Any

_COLUMNS = [
    "effect",
    "group",
    "level",
    "term",
    "estimate",
    "std.error",
    "statistic",
    "df",
    "p.value",
    "conf.low",
    "conf.high",
]
_VALID_EFFECTS = ("fixed", "ran_pars", "ran_vals")
_NAN = float("nan")


def tidy(
    model: MixedModelResult,
    effects: str | Sequence[str] = "fixed",
    *,
    conf_int: bool = False,
    conf_level: float = 0.95,
    ddf_method: str | None = "Satterthwaite",
) -> pd.DataFrame:
    """Return model components in a row-oriented table.

    Parameters
    ----------
    model : LmerResult, GlmerResult, or NlmerResult
        Fitted mixed model.
    effects : str or sequence of str, default "fixed"
        Components to include: ``"fixed"``, ``"ran_pars"`` (random-effect
        standard deviations and correlations), ``"ran_vals"`` (conditional
        modes), or ``"all"``.
    conf_int : bool, default False
        Include confidence limits for fixed effects.
    conf_level : float, default 0.95
        Confidence level between zero and one.
    ddf_method : str or None, default "Satterthwaite"
        Denominator degrees-of-freedom method for linear mixed models.
        Use ``"Kenward-Roger"`` or ``"normal"`` for alternatives. This
        argument is ignored for generalized and nonlinear models.

    Returns
    -------
    pandas.DataFrame
        A table with stable columns across model and effect types. Columns
        that do not apply to a row contain ``NaN``.
    """
    model_obj: Any = model
    selected = _normalize_effects(effects)
    if not 0.0 < conf_level < 1.0:
        raise ValueError("conf_level must be between 0 and 1")
    if not hasattr(model_obj, "fixef") or not hasattr(model_obj, "VarCorr"):
        raise TypeError("tidy() requires a fitted mixed-model result")

    rows: list[dict[str, Any]] = []
    if "fixed" in selected:
        rows.extend(_fixed_effect_rows(model_obj, conf_int, conf_level, ddf_method))
    if "ran_pars" in selected:
        rows.extend(_random_parameter_rows(model_obj))
    if "ran_vals" in selected:
        rows.extend(_random_value_rows(model_obj))

    return pd.DataFrame.from_records(rows, columns=_COLUMNS)


def glance(model: MixedModelResult) -> pd.DataFrame:
    """Return one row of model-level fit statistics.

    The result has the same schema for linear, generalized, and nonlinear
    mixed models, making it suitable for concatenating model comparisons.
    """
    model_obj: Any = model
    required = (
        "AIC",
        "BIC",
        "logLik",
        "nobs",
        "npar",
        "df_residual",
        "ngrps",
        "deviance",
        "converged",
        "n_iter",
    )
    if any(not hasattr(model_obj, name) for name in required):
        raise TypeError("glance() requires a fitted mixed-model result")

    log_lik = model_obj.logLik()
    log_lik_value = float(log_lik.value) if hasattr(log_lik, "value") else float(log_lik)
    group_counts = model_obj.ngrps()

    is_lmm = _model_flag(model_obj, "isLMM")
    is_glmm = _model_flag(model_obj, "isGLMM")
    is_nlmm = _model_flag(model_obj, "isNLMM")
    if is_glmm:
        family = type(model_obj.family).__name__
        model_type = "glmer"
    elif is_nlmm:
        family = "Gaussian"
        model_type = "nlmer"
    else:
        family = "Gaussian"
        model_type = "lmer" if is_lmm else type(model_obj).__name__

    is_reml = bool(model_obj.isREML()) if hasattr(model_obj, "isREML") else False
    singular = bool(model_obj.isSingular()) if hasattr(model_obj, "isSingular") else False

    row = {
        "model": model_type,
        "family": family,
        "nobs": int(model_obj.nobs()),
        "n_groups": int(sum(group_counts.values())),
        "n_grouping_factors": len(group_counts),
        "npar": int(model_obj.npar()),
        "df_residual": int(model_obj.df_residual()),
        "sigma": _model_sigma(model_obj),
        "log_lik": log_lik_value,
        "deviance": float(model_obj.deviance),
        "AIC": float(model_obj.AIC()),
        "BIC": float(model_obj.BIC()),
        "REML": is_reml,
        "converged": bool(model_obj.converged),
        "singular": singular,
        "n_iter": int(model_obj.n_iter),
    }
    return pd.DataFrame([row])


def _normalize_effects(effects: str | Sequence[str]) -> list[str]:
    requested = [effects] if isinstance(effects, str) else list(effects)
    if not requested:
        raise ValueError("effects must contain at least one component")

    normalized: list[str] = []
    for effect in requested:
        name = str(effect).strip().lower()
        if name == "all":
            for valid in _VALID_EFFECTS:
                if valid not in normalized:
                    normalized.append(valid)
        elif name in _VALID_EFFECTS:
            if name not in normalized:
                normalized.append(name)
        else:
            valid = "', '".join((*_VALID_EFFECTS, "all"))
            raise ValueError(f"Unknown effect '{effect}'. Use '{valid}'.")
    return normalized


def _fixed_effect_rows(
    model: Any,
    conf_int: bool,
    conf_level: float,
    ddf_method: str | None,
) -> list[dict[str, Any]]:
    fixed = model.fixef()
    names = list(fixed)
    estimates = np.asarray(list(fixed.values()), dtype=np.float64)
    covariance = np.asarray(model.vcov(), dtype=np.float64)
    if covariance.shape != (len(names), len(names)):
        raise ValueError(
            f"vcov() returned shape {covariance.shape}; expected {(len(names), len(names))}"
        )

    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    statistics = np.divide(
        estimates,
        standard_errors,
        out=np.full_like(estimates, np.nan),
        where=standard_errors > 0,
    )
    dfs, p_values = _fixed_effect_inference(model, names, statistics, ddf_method)

    if conf_int:
        alpha = 1.0 - conf_level
        critical = np.where(
            np.isfinite(dfs),
            stats.t.ppf(1.0 - alpha / 2.0, dfs),
            stats.norm.ppf(1.0 - alpha / 2.0),
        )
        lower = estimates - critical * standard_errors
        upper = estimates + critical * standard_errors
    else:
        lower = np.full_like(estimates, np.nan)
        upper = np.full_like(estimates, np.nan)

    return [
        _row(
            effect="fixed",
            term=name,
            estimate=estimates[index],
            standard_error=standard_errors[index],
            statistic=statistics[index],
            df=dfs[index],
            p_value=p_values[index],
            conf_low=lower[index],
            conf_high=upper[index],
        )
        for index, name in enumerate(names)
    ]


def _fixed_effect_inference(
    model: Any,
    names: list[str],
    statistics: np.ndarray,
    ddf_method: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    if _model_flag(model, "isLMM"):
        method = "normal" if ddf_method is None else ddf_method.strip().lower().replace("_", "-")
        if method in ("normal", "z"):
            dfs = np.full(len(names), np.nan)
            return dfs, 2.0 * stats.norm.sf(np.abs(statistics))

        from mixedlm.inference.ddf import kenward_roger_df, satterthwaite_df

        if method in ("satterthwaite", "satt"):
            ddf = satterthwaite_df(model)
        elif method in ("kenward-roger", "kr"):
            ddf = kenward_roger_df(model)
        else:
            raise ValueError(
                f"Unknown ddf_method '{ddf_method}'. Use 'Satterthwaite', "
                "'Kenward-Roger', or 'normal'."
            )
        dfs = np.asarray([ddf[name] for name in names], dtype=np.float64)
        return dfs, 2.0 * stats.t.sf(np.abs(statistics), dfs)

    if _model_flag(model, "isNLMM"):
        dfs = np.full(len(names), float(model.df_residual()))
        return dfs, 2.0 * stats.t.sf(np.abs(statistics), dfs)

    dfs = np.full(len(names), np.nan)
    return dfs, 2.0 * stats.norm.sf(np.abs(statistics))


def _random_parameter_rows(model: Any) -> list[dict[str, Any]]:
    if _model_flag(model, "isNLMM"):
        return _nlmm_random_parameter_rows(model)

    var_corr = model.VarCorr()
    rows: list[dict[str, Any]] = []

    for group_name, group in var_corr.groups.items():
        if hasattr(group, "term_names"):
            term_names = list(group.term_names)
            covariance = np.asarray(group.cov, dtype=np.float64)
            standard_deviations = np.sqrt(np.maximum(np.diag(covariance), 0.0))
            for index, term in enumerate(term_names):
                rows.append(
                    _row(
                        effect="ran_pars",
                        group=group_name,
                        term=f"sd__{term}",
                        estimate=standard_deviations[index],
                    )
                )
            correlation = getattr(group, "corr", None)
            if correlation is not None:
                correlation = np.asarray(correlation, dtype=np.float64)
                for row_index in range(1, len(term_names)):
                    for column_index in range(row_index):
                        rows.append(
                            _row(
                                effect="ran_pars",
                                group=group_name,
                                term=(f"cor__{term_names[column_index]}.{term_names[row_index]}"),
                                estimate=correlation[row_index, column_index],
                            )
                        )
        else:
            for term, variance in group.items():
                rows.append(
                    _row(
                        effect="ran_pars",
                        group=group_name,
                        term=f"sd__{term}",
                        estimate=np.sqrt(max(float(variance), 0.0)),
                    )
                )

    if hasattr(var_corr, "residual"):
        rows.append(
            _row(
                effect="ran_pars",
                group="Residual",
                term="sd__Observation",
                estimate=np.sqrt(max(float(var_corr.residual), 0.0)),
            )
        )
    return rows


def _nlmm_random_parameter_rows(model: Any) -> list[dict[str, Any]]:
    from mixedlm.estimation.nlmm import _build_psi_matrix

    term_names = [model.model.param_names[index] for index in model.random_params]
    covariance = _build_psi_matrix(model.theta, len(term_names)) * model.sigma**2
    standard_deviations = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    rows = [
        _row(
            effect="ran_pars",
            group=model.group_var,
            term=f"sd__{term}",
            estimate=standard_deviations[index],
        )
        for index, term in enumerate(term_names)
    ]

    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = covariance / np.outer(standard_deviations, standard_deviations)
    for row_index in range(1, len(term_names)):
        for column_index in range(row_index):
            value = correlation[row_index, column_index]
            rows.append(
                _row(
                    effect="ran_pars",
                    group=model.group_var,
                    term=f"cor__{term_names[column_index]}.{term_names[row_index]}",
                    estimate=value if np.isfinite(value) else 0.0,
                )
            )
    rows.append(
        _row(
            effect="ran_pars",
            group="Residual",
            term="sd__Observation",
            estimate=model.sigma,
        )
    )
    return rows


def _random_value_rows(model: Any) -> list[dict[str, Any]]:
    random_result = model.ranef(condVar=True) if hasattr(model, "matrices") else model.ranef()

    if hasattr(random_result, "values") and not isinstance(random_result, dict):
        values = random_result.values
        cond_var = random_result.condVar or {}
    else:
        values = random_result
        cond_var = {}

    rows: list[dict[str, Any]] = []
    for group_name, terms in values.items():
        levels = _group_levels(model, group_name)
        group_cond_var = cond_var.get(group_name, {})
        for term, estimates in terms.items():
            estimates_array = np.asarray(estimates, dtype=np.float64)
            if len(levels) != len(estimates_array):
                raise ValueError(
                    f"Random-effect levels for '{group_name}' have length {len(levels)}; "
                    f"expected {len(estimates_array)}"
                )
            variances = group_cond_var.get(term)
            if variances is None:
                standard_errors = np.full(len(estimates_array), np.nan)
            else:
                standard_errors = np.sqrt(np.maximum(np.asarray(variances, dtype=np.float64), 0.0))
            for index, estimate in enumerate(estimates_array):
                rows.append(
                    _row(
                        effect="ran_vals",
                        group=group_name,
                        level=levels[index],
                        term=term,
                        estimate=estimate,
                        standard_error=standard_errors[index],
                    )
                )
    return rows


def _group_levels(model: Any, group_name: str) -> list[Any]:
    if hasattr(model, "matrices"):
        for structure in model.matrices.random_structures:
            if structure.grouping_factor == group_name:
                ordered_levels = sorted(structure.level_map.items(), key=lambda item: item[1])
                return [level for level, _index in ordered_levels]
    if getattr(model, "group_var", None) == group_name:
        return list(model.group_levels)
    raise ValueError(f"Could not determine levels for grouping factor '{group_name}'")


def _model_flag(model: Any, name: str) -> bool:
    flag = getattr(model, name, None)
    return bool(flag()) if callable(flag) else False


def _model_sigma(model: Any) -> float:
    sigma = getattr(model, "sigma", None)
    if sigma is not None:
        return float(sigma() if callable(sigma) else sigma)
    getter = getattr(model, "get_sigma", None)
    return float(getter()) if callable(getter) else 1.0


def _row(
    *,
    effect: str,
    term: str,
    estimate: float,
    group: Any = None,
    level: Any = None,
    standard_error: float = _NAN,
    statistic: float = _NAN,
    df: float = _NAN,
    p_value: float = _NAN,
    conf_low: float = _NAN,
    conf_high: float = _NAN,
) -> dict[str, Any]:
    return {
        "effect": effect,
        "group": group,
        "level": level,
        "term": term,
        "estimate": float(estimate),
        "std.error": float(standard_error),
        "statistic": float(statistic),
        "df": float(df),
        "p.value": float(p_value),
        "conf.low": float(conf_low),
        "conf.high": float(conf_high),
    }
