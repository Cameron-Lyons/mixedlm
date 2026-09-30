from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from mixedlm.formula.terms import InteractionTerm, PowerTerm, VariableTerm
from mixedlm.matrices.design import build_fixed_matrix

if TYPE_CHECKING:
    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult


_MAX_EFFECT_MATRIX_ELEMENTS = 1_000_000


def _cartesian_grid(terms: list[str], values: list[list[Any]]) -> pd.DataFrame:
    """Build product columns in the same order as itertools.product."""
    n_rows = math.prod(map(len, values))
    columns = {}
    stride = n_rows
    for term, levels in zip(terms, values, strict=True):
        stride //= len(levels)
        indices = np.tile(
            np.repeat(np.arange(len(levels)), stride), n_rows // (len(levels) * stride)
        )
        columns[term] = pd.Series(levels).array.take(indices)
    return pd.DataFrame(columns, copy=False)


def _prediction_moments(
    grid: pd.DataFrame,
    beta: NDArray[np.float64],
    build_matrix: Callable[[pd.DataFrame], NDArray[np.floating]],
    covariance: Callable[[], NDArray[np.floating]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build and project one batch of design rows at a time."""
    eta = np.empty(len(grid), dtype=np.float64)
    variance = np.empty(len(grid), dtype=np.float64)
    batch_rows = max(1, _MAX_EFFECT_MATRIX_ELEMENTS // max(1, len(beta), len(grid.columns)))
    vcov = None
    for start in range(0, len(grid), batch_rows):
        rows = slice(start, start + batch_rows)
        matrix = build_matrix(grid.iloc[rows])
        if vcov is None:
            # Validate the design (including contrasts) before requesting covariance.
            vcov = np.asarray(covariance(), dtype=np.float64)
        eta[rows] = matrix @ beta
        projected = matrix @ vcov
        variance[rows] = np.einsum("ij,ij->i", projected, matrix)
        del matrix, projected
    return eta, np.sqrt(np.maximum(variance, 0.0))


def _as_pandas_frame(frame: Any) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    if "polars" in type(frame).__module__ and hasattr(frame, "to_dict"):
        result = pd.DataFrame(frame.to_dict(as_series=False))
        for name in frame.columns:
            column = frame.get_column(name)
            if "Categorical" in str(column.dtype) or "Enum" in str(column.dtype):
                categories = column.cat.get_categories().to_list()
                result[name] = pd.Categorical(result[name], categories=categories)
        return result
    raise TypeError(f"Expected a pandas or Polars model frame, got {type(frame).__name__}")


def _fixed_variable_order(model: LmerResult | GlmerResult) -> list[str]:
    variables: list[str] = []
    for term in model.formula.fixed.terms:
        candidates: tuple[str, ...]
        if isinstance(term, VariableTerm | PowerTerm):
            candidates = (term.name,)
        elif isinstance(term, InteractionTerm):
            candidates = term.source_variables
        else:
            continue
        for name in candidates:
            if name not in variables:
                variables.append(name)
    return variables


def _is_factor(series: pd.Series) -> bool:
    return bool(
        isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(series.dtype)
        or pd.api.types.is_string_dtype(series.dtype)
    )


def _factor_levels(series: pd.Series) -> list[Any]:
    if isinstance(series.dtype, pd.CategoricalDtype):
        return series.cat.categories.tolist()
    return sorted(series.dropna().unique().tolist())


def _coerce_values(value: Any, name: str) -> list[Any]:
    if isinstance(value, str) or np.isscalar(value):
        values = [value]
    else:
        try:
            values = list(value)
        except TypeError as exc:
            raise TypeError(f"Values for '{name}' must be a scalar or iterable") from exc
    if not values:
        raise ValueError(f"Values for '{name}' cannot be empty")
    return values


def _validate_values(series: pd.Series, values: list[Any]) -> list[Any]:
    if _is_factor(series):
        levels = _factor_levels(series)
        unknown = [value for value in values if value not in levels]
        if unknown:
            raise ValueError(
                f"Unknown level(s) for '{series.name}': {unknown}. Available levels: {levels}"
            )
        return values
    try:
        numeric = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Values for numeric variable '{series.name}' must be numeric") from exc
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"Values for numeric variable '{series.name}' must be finite")
    return numeric.tolist()


def _default_values(series: pd.Series, n_points: int) -> list[Any]:
    clean = series.dropna()
    if clean.empty:
        raise ValueError(f"Variable '{series.name}' has no non-missing values")
    if _is_factor(series):
        return _factor_levels(series)

    unique = np.sort(clean.unique())
    if len(unique) <= n_points:
        return unique.tolist()
    return np.linspace(float(clean.min()), float(clean.max()), n_points).tolist()


def _reference_value(series: pd.Series) -> Any:
    clean = series.dropna()
    if clean.empty:
        raise ValueError(f"Variable '{series.name}' has no non-missing values")
    if _is_factor(series):
        return _factor_levels(series)[0]
    return float(clean.mean())


def _normalize_terms(terms: str | Sequence[str]) -> list[str]:
    result = [terms] if isinstance(terms, str) else list(terms)
    if not result:
        raise ValueError("terms must contain at least one fixed-effect variable")
    if any(not isinstance(term, str) or not term for term in result):
        raise TypeError("terms must contain non-empty variable names")
    if len(set(result)) != len(result):
        raise ValueError("terms must not contain duplicates")
    return result


def _prediction_grid(
    model: LmerResult | GlmerResult,
    terms: list[str],
    at: Mapping[str, Any],
    n_points: int,
) -> pd.DataFrame:
    frame_source = model.matrices.frame
    if frame_source is None:
        frame_source = model.model_frame()
    frame = _as_pandas_frame(frame_source)
    fixed_variables = _fixed_variable_order(model)
    available = set(fixed_variables)

    missing = [term for term in terms if term not in available]
    if missing:
        raise ValueError(
            f"Unknown fixed-effect variable(s): {', '.join(missing)}. "
            f"Available variables: {', '.join(fixed_variables)}"
        )
    unknown_at = [name for name in at if name not in available]
    if unknown_at:
        raise ValueError(f"Unknown variable(s) in at: {', '.join(unknown_at)}")

    grid_values: list[list[Any]] = []
    for term in terms:
        values = (
            _coerce_values(at[term], term) if term in at else _default_values(frame[term], n_points)
        )
        grid_values.append(_validate_values(frame[term], values))

    references = {}
    for variable in fixed_variables:
        if variable in terms:
            continue
        if variable in at:
            values = _validate_values(frame[variable], _coerce_values(at[variable], variable))
            if len(values) != 1:
                raise ValueError(
                    f"Non-focal variable '{variable}' must have one value in at; "
                    "include it in terms to predict a grid"
                )
            references[variable] = values[0]
        else:
            references[variable] = _reference_value(frame[variable])

    grid = _cartesian_grid(terms, grid_values)
    for variable, value in references.items():
        grid[variable] = value

    for variable in fixed_variables:
        source = frame[variable]
        if _is_factor(source):
            levels = _factor_levels(source)
            ordered = bool(isinstance(source.dtype, pd.CategoricalDtype) and source.dtype.ordered)
            grid[variable] = pd.Categorical(grid[variable], categories=levels, ordered=ordered)

    return grid


def _effect_design_matrix(
    model: LmerResult | GlmerResult,
    grid: pd.DataFrame,
    contrasts: dict[str, str | NDArray[np.floating]] | None,
) -> NDArray[np.float64]:
    X_grid, fixed_names = build_fixed_matrix(model.formula, grid, contrasts=contrasts)
    column_indices = {name: index for index, name in enumerate(fixed_names)}
    try:
        fitted_indices = [column_indices[name] for name in model.matrices.fixed_names]
    except KeyError as exc:
        raise ValueError(
            f"Prediction grid is missing fitted fixed-effect column '{exc.args[0]}'"
        ) from None
    return np.asarray(X_grid[:, fitted_indices], dtype=np.float64)


def ggpredict(
    model: LmerResult | GlmerResult,
    terms: str | Sequence[str],
    *,
    at: Mapping[str, Any] | None = None,
    type: str = "response",
    level: float = 0.95,
    n_points: int = 25,
    offset: float = 0.0,
    contrasts: dict[str, str | NDArray[np.floating]] | None = None,
) -> pd.DataFrame:
    """Compute adjusted fixed-effect predictions over a compact value grid.

    Numeric variables not in ``terms`` are held at their mean, while categorical
    variables are held at their reference level. Confidence intervals account for
    the complete fixed-effect covariance matrix. GLMM intervals are constructed on
    the link scale and transformed to the response scale when requested.

    Design matrices and covariance projections are evaluated in batches. The
    complete reference grid and returned predictions remain in memory.

    Parameters
    ----------
    model : LmerResult or GlmerResult
        A fitted linear or generalized linear mixed model.
    terms : str or sequence of str
        Fixed-effect variables that define the prediction grid.
    at : mapping, optional
        Explicit values for focal variables or a single conditioning value for
        non-focal variables.
    type : {"response", "link"}, default "response"
        Prediction scale. The two scales are identical for linear mixed models.
    level : float, default 0.95
        Confidence level.
    n_points : int, default 25
        Maximum number of automatically generated values per numeric variable.
    offset : float, default 0.0
        Constant offset added to the linear predictor.
    contrasts : dict, optional
        Contrast specification used when fitting the model. Supply this for
        non-default categorical contrasts.

    Returns
    -------
    pandas.DataFrame
        Grid variables followed by ``predicted``, ``std.error``, ``conf.low``,
        and ``conf.high`` columns.
    """
    if type not in {"response", "link"}:
        raise ValueError("type must be 'response' or 'link'")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be between 0 and 1")
    if isinstance(n_points, bool) or not isinstance(n_points, int) or n_points < 2:
        raise ValueError("n_points must be an integer of at least 2")
    try:
        offset_value = float(offset)
    except (TypeError, ValueError) as exc:
        raise TypeError("offset must be a finite scalar") from exc
    if not np.isfinite(offset_value):
        raise ValueError("offset must be finite")
    if not hasattr(model, "formula") or not hasattr(model, "matrices"):
        raise TypeError("model must be a fitted linear or generalized linear mixed model")

    normalized_terms = _normalize_terms(terms)
    grid = _prediction_grid(model, normalized_terms, at or {}, n_points)

    beta = np.asarray(model.beta, dtype=np.float64)
    eta, se_eta = _prediction_moments(
        grid, beta, lambda chunk: _effect_design_matrix(model, chunk, contrasts), model.vcov
    )
    eta += offset_value

    is_glmm = bool(hasattr(model, "isGLMM") and model.isGLMM())
    if is_glmm:
        critical = float(stats.norm.ppf(1.0 - (1.0 - level) / 2.0))
    else:
        df = float(model.df_residual())
        critical = float(stats.t.ppf(1.0 - (1.0 - level) / 2.0, df))

    lower_eta = eta - critical * se_eta
    upper_eta = eta + critical * se_eta

    if is_glmm and type == "response":
        family = getattr(model, "family", None)
        if family is None:
            raise TypeError("Generalized linear mixed model must define a family")
        predicted = np.asarray(family.link.inverse(eta), dtype=np.float64)
        lower_response = np.asarray(family.link.inverse(lower_eta), dtype=np.float64)
        upper_response = np.asarray(family.link.inverse(upper_eta), dtype=np.float64)
        lower = np.minimum(lower_response, upper_response)
        upper = np.maximum(lower_response, upper_response)
        link_derivative = np.asarray(family.link.deriv(predicted), dtype=np.float64)
        standard_error = se_eta / np.maximum(np.abs(link_derivative), np.finfo(float).tiny)
    else:
        predicted = eta
        standard_error = se_eta
        lower = lower_eta
        upper = upper_eta

    result = grid[normalized_terms].copy()
    result["predicted"] = predicted
    result["std.error"] = standard_error
    result["conf.low"] = lower
    result["conf.high"] = upper
    result.attrs.update(
        {
            "type": type,
            "level": level,
            "offset": offset_value,
            "adjustment": "numeric means and categorical reference levels",
        }
    )
    return result


def allEffects(
    model: LmerResult | GlmerResult,
    *,
    at: Mapping[str, Any] | None = None,
    type: str = "response",
    level: float = 0.95,
    n_points: int = 25,
    offset: float = 0.0,
    contrasts: dict[str, str | NDArray[np.floating]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute a one-variable adjusted prediction grid for every fixed effect."""
    variables = _fixed_variable_order(model)
    return {
        variable: ggpredict(
            model,
            variable,
            at=at,
            type=type,
            level=level,
            n_points=n_points,
            offset=offset,
            contrasts=contrasts,
        )
        for variable in variables
    }
