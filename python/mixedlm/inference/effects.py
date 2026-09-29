from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from mixedlm.formula.terms import InteractionTerm, PowerTerm, VariableTerm

if TYPE_CHECKING:
    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult


def _as_pandas_frame(frame: Any) -> pd.DataFrame:
    """Access the model frame for read-only reference-value calculations."""
    if isinstance(frame, pd.DataFrame):
        return frame
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


def _validate_values(
    series: pd.Series, values: list[Any], levels: list[Any] | None = None
) -> list[Any]:
    if levels is not None or _is_factor(series):
        levels = _factor_levels(series) if levels is None else levels
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


def _default_values(series: pd.Series, n_points: int, levels: list[Any] | None = None) -> list[Any]:
    if levels is not None:
        return levels
    clean = series.dropna()
    if clean.empty:
        raise ValueError(f"Variable '{series.name}' has no non-missing values")
    if _is_factor(series):
        return _factor_levels(series)

    unique = clean.unique()
    if len(unique) <= n_points:
        return np.sort(unique).tolist()
    return np.linspace(float(clean.min()), float(clean.max()), n_points).tolist()


def _reference_value(series: pd.Series, levels: list[Any] | None = None) -> Any:
    if levels is not None:
        return levels[0]
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


class _EffectGrid:
    """Prepare reference values once, then build independent prediction grids."""

    def __init__(
        self,
        model: LmerResult | GlmerResult,
        at: Mapping[str, Any] | None,
        n_points: int,
    ) -> None:
        self.model = model
        frame_source = model.matrices.frame
        if frame_source is None:
            frame_source = model.model_frame()
        self.frame = _as_pandas_frame(frame_source)
        self.variables = _fixed_variable_order(model)
        self.n_points = n_points
        overrides = {} if at is None else at
        if not isinstance(overrides, Mapping):
            raise TypeError("at must be a mapping of fixed-effect variables to values")
        unknown_at = [name for name in overrides if name not in self.variables]
        if unknown_at:
            raise ValueError(f"Unknown variable(s) in at: {', '.join(map(str, unknown_at))}")

        self.categories: dict[str, pd.CategoricalDtype] = {}
        for variable in self.variables:
            source = self.frame[variable]
            levels = model.matrices.category_levels.get(variable)
            if levels is not None or _is_factor(source):
                levels = _factor_levels(source) if levels is None else levels
                ordered = bool(
                    isinstance(source.dtype, pd.CategoricalDtype) and source.dtype.ordered
                )
                self.categories[variable] = pd.CategoricalDtype(levels, ordered=ordered)
        self.overrides = {
            name: _validate_values(
                self.frame[name], _coerce_values(values, name), self._levels(name)
            )
            for name, values in overrides.items()
        }
        self.references: dict[str, Any] = {}

    def _levels(self, variable: str) -> list[Any] | None:
        dtype = self.categories.get(variable)
        return None if dtype is None else dtype.categories.tolist()

    def validate_terms(self, terms: list[str]) -> None:
        missing = [term for term in terms if term not in self.variables]
        if missing:
            raise ValueError(
                f"Unknown fixed-effect variable(s): {', '.join(missing)}. "
                f"Available variables: {', '.join(self.variables)}"
            )
        for variable, values in self.overrides.items():
            if variable not in terms and len(values) != 1:
                raise ValueError(
                    f"Non-focal variable '{variable}' must have one value in at; "
                    "include it in terms to predict a grid"
                )

    def build(
        self,
        terms: list[str],
        contrasts: dict[str, str | NDArray[np.floating]] | None,
    ) -> tuple[pd.DataFrame, NDArray[np.float64]]:
        grid_values = []
        for term in terms:
            levels = self._levels(term)
            values = self.overrides.get(term)
            if values is None:
                values = _validate_values(
                    self.frame[term],
                    _default_values(self.frame[term], self.n_points, levels),
                    levels,
                )
            grid_values.append(values)
        grid = pd.DataFrame(itertools.product(*grid_values), columns=terms)
        for variable in self.variables:
            if variable in terms:
                continue
            if variable in self.overrides:
                value = self.overrides[variable][0]
            else:
                if variable not in self.references:
                    self.references[variable] = _reference_value(
                        self.frame[variable], self._levels(variable)
                    )
                value = self.references[variable]
            grid[variable] = value
        for variable, dtype in self.categories.items():
            grid[variable] = pd.Categorical(grid[variable], dtype=dtype)
        matrix = self.model._prediction_fixed_matrix(grid, contrasts=contrasts)
        return grid, np.asarray(matrix, dtype=np.float64)


def _validate_prediction_options(
    model: LmerResult | GlmerResult,
    type: str,
    level: float,
    n_points: int,
    offset: float,
) -> float:
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
    return offset_value


class _EffectPrediction:
    """Share coefficient uncertainty and display settings within one request."""

    def __init__(
        self, model: LmerResult | GlmerResult, type: str, level: float, offset: float
    ) -> None:
        self.beta = np.asarray(model.beta, dtype=np.float64)
        self.vcov = np.asarray(model.vcov(), dtype=np.float64)
        self.type = type
        self.level = level
        self.offset = offset
        is_glmm = bool(hasattr(model, "isGLMM") and model.isGLMM())
        self.family = getattr(model, "family", None) if is_glmm else None
        if is_glmm and type == "response" and self.family is None:
            raise TypeError("Generalized linear mixed model must define a family")
        self.critical = float(
            stats.norm.ppf(1.0 - (1.0 - level) / 2.0)
            if is_glmm
            else stats.t.ppf(1.0 - (1.0 - level) / 2.0, float(model.df_residual()))
        )

    def predict(
        self, terms: list[str], grid: pd.DataFrame, matrix: NDArray[np.float64]
    ) -> pd.DataFrame:
        eta = matrix @ self.beta + self.offset
        variance = np.einsum("ij,jk,ik->i", matrix, self.vcov, matrix, optimize=True)
        se_eta = np.sqrt(np.maximum(variance, 0.0))
        lower_eta = eta - self.critical * se_eta
        upper_eta = eta + self.critical * se_eta

        if self.family is not None and self.type == "response":
            predicted = np.asarray(self.family.link.inverse(eta), dtype=np.float64)
            lower_response = np.asarray(self.family.link.inverse(lower_eta), dtype=np.float64)
            upper_response = np.asarray(self.family.link.inverse(upper_eta), dtype=np.float64)
            lower = np.minimum(lower_response, upper_response)
            upper = np.maximum(lower_response, upper_response)
            link_derivative = np.asarray(self.family.link.deriv(predicted), dtype=np.float64)
            standard_error = se_eta / np.maximum(np.abs(link_derivative), np.finfo(float).tiny)
        else:
            predicted = eta
            standard_error = se_eta
            lower = lower_eta
            upper = upper_eta

        result = grid[terms].copy()
        result["predicted"] = predicted
        result["std.error"] = standard_error
        result["conf.low"] = lower
        result["conf.high"] = upper
        result.attrs.update(
            {
                "type": self.type,
                "level": self.level,
                "offset": self.offset,
                "adjustment": "numeric means and categorical reference levels",
            }
        )
        return result


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
        Defaults to the fitted categorical contrasts. An explicit mapping
        overrides them and should match the fitted coefficient parameterization.

    Returns
    -------
    pandas.DataFrame
        Grid variables followed by ``predicted``, ``std.error``, ``conf.low``,
        and ``conf.high`` columns.
    """
    offset_value = _validate_prediction_options(model, type, level, n_points, offset)
    normalized_terms = _normalize_terms(terms)
    builder = _EffectGrid(model, at, n_points)
    builder.validate_terms(normalized_terms)
    grid, matrix = builder.build(normalized_terms, contrasts)
    prediction = _EffectPrediction(model, type, level, offset_value)
    return prediction.predict(normalized_terms, grid, matrix)


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
    """Compute one grid per fixed-effect variable, sharing setup within this call.

    Frame conversion, conditioning values, coefficient covariance, and confidence
    cutoffs are reused. Each grid is evaluated separately, and nothing is cached
    on the fitted model by this function. With multiple fixed-effect variables,
    ``at`` must supply one value per variable because each also conditions the
    other grids.
    """
    offset_value = _validate_prediction_options(model, type, level, n_points, offset)
    builder = _EffectGrid(model, at, n_points)
    for variable in builder.variables:
        builder.validate_terms([variable])
    prediction = None
    results = {}
    for variable in builder.variables:
        grid, matrix = builder.build([variable], contrasts)
        if prediction is None:
            prediction = _EffectPrediction(model, type, level, offset_value)
        results[variable] = prediction.predict([variable], grid, matrix)
    return results
