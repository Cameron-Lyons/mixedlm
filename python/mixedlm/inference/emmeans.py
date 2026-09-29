from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from math import prod
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from mixedlm.utils.dataframe import get_categories, get_column_numpy, is_categorical_or_string

if TYPE_CHECKING:
    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult


def _rowwise_quadratic_form(
    coefficients: NDArray[np.floating],
    covariance: NDArray[np.floating],
) -> NDArray[np.floating]:
    projected = coefficients @ covariance
    return np.einsum("ij,ij->i", projected, coefficients)


@dataclass
class EmmeanResult:
    emmean: NDArray[np.floating]
    se: NDArray[np.floating]
    df: float
    lower: NDArray[np.floating]
    upper: NDArray[np.floating]
    grid: pd.DataFrame
    level: float

    def __str__(self) -> str:
        lines = []
        lines.append("Estimated Marginal Means")
        lines.append("")

        col_widths = {}
        for col in self.grid.columns:
            max_width = max(len(str(col)), max(len(str(v)) for v in self.grid[col]))
            col_widths[col] = max(max_width, 8)

        header = ""
        for col in self.grid.columns:
            header += f"{col:>{col_widths[col]}} "
        header += f"{'emmean':>10} {'SE':>8} {'df':>6} {'lower':>10} {'upper':>10}"
        lines.append(header)

        for i in range(len(self.emmean)):
            row = ""
            for col in self.grid.columns:
                val = self.grid.iloc[i][col]
                row += f"{str(val):>{col_widths[col]}} "
            row += f"{self.emmean[i]:>10.3f} {self.se[i]:>8.3f} {self.df:>6.1f}"
            row += f" {self.lower[i]:>10.3f} {self.upper[i]:>10.3f}"
            lines.append(row)

        lines.append("")
        lines.append(f"Confidence level: {self.level:.0%}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"EmmeanResult(n={len(self.emmean)}, level={self.level})"


@dataclass
class ContrastResult:
    contrast: list[str]
    estimate: NDArray[np.floating]
    se: NDArray[np.floating]
    df: float
    t_ratio: NDArray[np.floating]
    p_value: NDArray[np.floating]
    adjust: str
    grid: pd.DataFrame | None = None

    def __str__(self) -> str:
        lines = []
        lines.append("Pairwise Comparisons")
        lines.append("")

        max_contrast_len = max(len(c) for c in self.contrast)
        max_contrast_len = max(max_contrast_len, 8)

        header = f"{'contrast':<{max_contrast_len}} {'estimate':>10} {'SE':>8}"
        header += f" {'df':>6} {'t.ratio':>8} {'p.value':>10}"
        lines.append(header)

        for i in range(len(self.contrast)):
            p_str = f"{self.p_value[i]:.4f}" if self.p_value[i] >= 0.0001 else "<.0001"
            row = f"{self.contrast[i]:<{max_contrast_len}} {self.estimate[i]:>10.3f}"
            row += f" {self.se[i]:>8.3f} {self.df:>6.1f} {self.t_ratio[i]:>8.3f}"
            row += f" {p_str:>10}"
            lines.append(row)

        lines.append("")
        if self.adjust != "none":
            lines.append(f"P-value adjustment: {self.adjust}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ContrastResult(n={len(self.contrast)}, adjust='{self.adjust}')"


@dataclass
class Emmeans:
    result: EmmeanResult
    _L: NDArray[np.floating]
    _vcov: NDArray[np.floating]
    _beta: NDArray[np.floating]
    _df: float
    _specs: list[str]
    _levels: list[list[Any]]
    _by: list[str] = field(default_factory=list)

    def _grouped_contrasts(self, compute: Callable[[Emmeans], ContrastResult]) -> ContrastResult:
        """Compare means and adjust p-values independently within each by group."""
        results: list[ContrastResult] = []
        grids: list[pd.DataFrame] = []
        labels: list[str] = []
        groups = self.result.grid.groupby(self._by, sort=False, observed=True, dropna=False)
        for indices in groups.indices.values():
            grid = self.result.grid.iloc[indices].reset_index(drop=True)
            subset = replace(
                self.result,
                emmean=self.result.emmean[indices],
                se=self.result.se[indices],
                lower=self.result.lower[indices],
                upper=self.result.upper[indices],
                grid=grid,
            )
            means = replace(self, result=subset, _L=self._L[indices], _by=[])
            result = compute(means)
            values = {name: grid[name].iloc[0] for name in self._by}
            description = ", ".join(f"{name}={value}" for name, value in values.items())
            labels.extend(f"{label} | {description}" for label in result.contrast)
            grids.append(pd.DataFrame([values] * len(result.contrast), columns=self._by))
            results.append(result)

        return ContrastResult(
            contrast=labels,
            estimate=np.concatenate([result.estimate for result in results]),
            se=np.concatenate([result.se for result in results]),
            df=self._df,
            t_ratio=np.concatenate([result.t_ratio for result in results]),
            p_value=np.concatenate([result.p_value for result in results]),
            adjust=results[0].adjust,
            grid=pd.concat(grids, ignore_index=True),
        )

    def pairs(
        self,
        adjust: str = "tukey",
        level: float = 0.95,
    ) -> ContrastResult:
        if self._by:
            return self._grouped_contrasts(lambda means: means.pairs(adjust=adjust, level=level))
        n_levels = len(self.result.emmean)
        if n_levels < 2:
            raise ValueError("Need at least 2 levels for pairwise comparisons")

        grid_labels = []
        for i in range(n_levels):
            parts = []
            for spec in self._specs:
                parts.append(str(self.result.grid.iloc[i][spec]))
            grid_labels.append(",".join(parts))

        left_indices, right_indices = np.triu_indices(n_levels, k=1)
        contrast_labels = [
            f"{grid_labels[i]} - {grid_labels[j]}"
            for i, j in zip(left_indices, right_indices, strict=True)
        ]
        L_contrast = self._L[left_indices] - self._L[right_indices]

        estimates = L_contrast @ self._beta
        var_contrast = _rowwise_quadratic_form(L_contrast, self._vcov)
        se_contrast = np.sqrt(np.maximum(var_contrast, 0))

        t_ratio = estimates / se_contrast

        raw_p = 2 * (1 - stats.t.cdf(np.abs(t_ratio), self._df))
        p_adjusted = _adjust_pvalues(raw_p, adjust, n_levels, self._df, t_ratio)

        return ContrastResult(
            contrast=contrast_labels,
            estimate=estimates,
            se=se_contrast,
            df=self._df,
            t_ratio=t_ratio,
            p_value=p_adjusted,
            adjust=adjust,
        )

    def contrast(
        self,
        method: str | NDArray[np.floating] = "pairwise",
        adjust: str = "none",
        level: float = 0.95,
    ) -> ContrastResult:
        if isinstance(method, str):
            if method == "pairwise":
                return self.pairs(adjust=adjust if adjust != "none" else "tukey", level=level)
            elif method == "trt.vs.ctrl":
                return self._trt_vs_ctrl(adjust=adjust, level=level)
            else:
                raise ValueError(f"Unknown contrast method: {method}")
        else:
            return self._custom_contrast(method, adjust=adjust, level=level)

    def _trt_vs_ctrl(
        self,
        ctrl_idx: int = 0,
        adjust: str = "dunnett",
        level: float = 0.95,
    ) -> ContrastResult:
        if self._by:
            return self._grouped_contrasts(
                lambda means: means._trt_vs_ctrl(ctrl_idx=ctrl_idx, adjust=adjust, level=level)
            )
        n_levels = len(self.result.emmean)

        grid_labels = []
        for i in range(n_levels):
            parts = [str(self.result.grid.iloc[i][spec]) for spec in self._specs]
            grid_labels.append(",".join(parts))

        treatment_indices = np.delete(np.arange(n_levels), ctrl_idx)
        contrast_labels = [f"{grid_labels[i]} - {grid_labels[ctrl_idx]}" for i in treatment_indices]
        L_contrast = self._L[treatment_indices] - self._L[ctrl_idx]
        estimates = L_contrast @ self._beta
        var_contrast = _rowwise_quadratic_form(L_contrast, self._vcov)
        se_contrast = np.sqrt(np.maximum(var_contrast, 0))
        t_ratio = estimates / se_contrast
        raw_p = 2 * (1 - stats.t.cdf(np.abs(t_ratio), self._df))
        p_adjusted = _adjust_pvalues(raw_p, adjust, n_levels, self._df, t_ratio)

        return ContrastResult(
            contrast=contrast_labels,
            estimate=estimates,
            se=se_contrast,
            df=self._df,
            t_ratio=t_ratio,
            p_value=p_adjusted,
            adjust=adjust,
        )

    def _custom_contrast(
        self,
        C: NDArray[np.floating],
        adjust: str = "none",
        level: float = 0.95,
    ) -> ContrastResult:
        if self._by:
            return self._grouped_contrasts(
                lambda means: means._custom_contrast(C, adjust=adjust, level=level)
            )
        n_contrasts = C.shape[0]
        L_contrast = C @ self._L
        estimates = L_contrast @ self._beta
        var_contrast = _rowwise_quadratic_form(L_contrast, self._vcov)
        se_contrast = np.sqrt(np.maximum(var_contrast, 0))
        t_ratio = estimates / se_contrast
        raw_p = 2 * (1 - stats.t.cdf(np.abs(t_ratio), self._df))
        p_adjusted = _adjust_pvalues(raw_p, adjust, n_contrasts, self._df, t_ratio)

        contrast_labels = [f"C{i + 1}" for i in range(n_contrasts)]

        return ContrastResult(
            contrast=contrast_labels,
            estimate=estimates,
            se=se_contrast,
            df=self._df,
            t_ratio=t_ratio,
            p_value=p_adjusted,
            adjust=adjust,
        )

    def __str__(self) -> str:
        return str(self.result)

    def __repr__(self) -> str:
        return f"Emmeans(specs={self._specs}, n={len(self.result.emmean)})"


def _adjust_pvalues(
    p: NDArray[np.floating],
    method: str,
    n_groups: int,
    df: float,
    t_ratio: NDArray[np.floating] | None = None,
) -> NDArray[np.floating]:
    if method == "none":
        return p
    elif method == "bonferroni":
        return np.minimum(p * len(p), 1.0)
    elif method == "holm":
        n = len(p)
        sorted_idx = np.argsort(p)
        sorted_p = p[sorted_idx]
        adjusted = np.zeros(n)
        cummax = 0.0
        for i, idx in enumerate(sorted_idx):
            adj_p = sorted_p[i] * (n - i)
            cummax = max(cummax, adj_p)
            adjusted[idx] = min(cummax, 1.0)
        return adjusted
    elif method == "fdr":
        n = len(p)
        sorted_idx = np.argsort(p)[::-1]
        sorted_p = p[sorted_idx]
        adjusted = np.zeros(n)
        cummin = 1.0
        for i, idx in enumerate(sorted_idx):
            rank = n - i
            adj_p = sorted_p[i] * n / rank
            cummin = min(cummin, adj_p)
            adjusted[idx] = min(cummin, 1.0)
        return adjusted
    elif method == "tukey":
        if t_ratio is None:
            return p
        q = np.abs(t_ratio) * np.sqrt(2)
        return stats.studentized_range.sf(q, n_groups, df)
    elif method == "dunnett":
        return np.minimum(p * (n_groups - 1), 1.0)
    else:
        return p


def _predictor_names(names: str | list[str], argument: str) -> list[str]:
    if isinstance(names, str):
        return [names]
    if not isinstance(names, (list, tuple)) or any(not isinstance(name, str) for name in names):
        raise TypeError(f"{argument} must be a predictor name or a sequence of names")
    if len(set(names)) != len(names):
        raise ValueError(f"{argument} must not contain duplicate predictor names")
    return list(names)


def _grid_values(values: Any, name: str, *, numeric: bool) -> list[Any]:
    array = np.asarray(values, dtype=None if numeric else object)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"Reference values for '{name}' must be a nonempty scalar or 1-D sequence")
    if pd.isna(array).any():
        raise ValueError(f"Reference values for '{name}' must not contain missing values")
    if numeric:
        try:
            array = array.astype(np.float64)
        except (TypeError, ValueError):
            raise ValueError(f"Reference values for '{name}' must be numeric") from None
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Reference values for '{name}' must be finite")
    if pd.Index(array).has_duplicates:
        raise ValueError(f"Reference values for '{name}' must be distinct")
    return array.tolist()


def _reference_levels(
    model: LmerResult | GlmerResult,
    at: dict[str, Any] | None,
    cov_reduce: Callable[[pd.Series], float],
) -> dict[str, list[Any]]:
    frame = model.model_frame()
    variables = model.terms().fixed_variables
    if at is not None and not isinstance(at, dict):
        raise TypeError("at must be a dictionary of predictor names and reference values")
    overrides = {} if at is None else at
    if any(not isinstance(name, str) for name in overrides):
        raise TypeError("Predictor names in at must be strings")
    unknown = set(overrides) - set(variables)
    if unknown:
        raise ValueError(f"Unknown fixed-effect predictor(s) in at: {sorted(unknown)}")

    levels = {}
    for name in sorted(variables):
        categorical = is_categorical_or_string(frame, name)
        if name in overrides:
            values = overrides[name]
        elif categorical:
            values = get_categories(frame, name)
        else:
            column = (
                frame[name]
                if isinstance(frame, pd.DataFrame)
                else pd.Series(get_column_numpy(frame, name), name=name)
            )
            values = cov_reduce(column)
        levels[name] = _grid_values(values, name, numeric=not categorical)
    return levels


def emmeans(
    model: LmerResult | GlmerResult,
    specs: str | list[str],
    by: str | list[str] | None = None,
    at: dict[str, Any] | None = None,
    cov_reduce: Callable[[pd.Series], float] = np.mean,
    type: str = "response",
    level: float = 0.95,
    *,
    _by: str | list[str] | None = None,
) -> Emmeans:
    """Estimate means over a reference grid, optionally comparing within by groups.

    Numeric predictors use ``cov_reduce`` unless ``at`` supplies reference values.
    Predictors in ``specs`` or ``by`` identify result rows; all other reference
    dimensions are averaged with equal weights. Contrasts compare means within
    each ``by`` group and adjust that group's p-values separately.
    """
    if type not in {"link", "response"}:
        raise ValueError("type must be 'link' or 'response'")
    if not np.isfinite(level) or not 0 < level < 1:
        raise ValueError("level must be a finite number strictly between 0 and 1")
    if by is not None and _by is not None:
        raise ValueError("Specify only one of by and _by")
    if _by is not None:
        by = _by

    spec_names = _predictor_names(specs, "specs")
    by_names = _predictor_names([] if by is None else by, "by")
    result_names = spec_names + [name for name in by_names if name not in spec_names]
    levels = _reference_levels(model, at, cov_reduce)
    for name in result_names:
        if name not in levels:
            raise ValueError(
                f"Variable '{name}' must name a fixed-effect predictor. "
                f"Available predictors: {list(levels)}"
            )

    beta = model.beta
    df_resid = float(model.df_residual())
    averaged_names = [name for name in levels if name not in result_names]
    all_names = result_names + averaged_names
    grid = pd.DataFrame(itertools.product(*(levels[name] for name in all_names)), columns=all_names)
    X_grid = model._prediction_fixed_matrix(grid)
    result_grid = pd.DataFrame(
        itertools.product(*(levels[name] for name in result_names)), columns=result_names
    )
    n_averaged = prod(len(levels[name]) for name in averaged_names)
    L = X_grid.reshape(len(result_grid), n_averaged, len(beta)).mean(axis=1)
    vcov = model.vcov()

    em_values = L @ beta
    var_em = _rowwise_quadratic_form(L, vcov)
    se_em = np.sqrt(np.maximum(var_em, 0))

    family = getattr(model, "family", None)
    alpha = 1 - level

    if family is not None:
        critical_value = stats.norm.ppf(1 - alpha / 2)
    else:
        critical_value = stats.t.ppf(1 - alpha / 2, df_resid)

    lower = em_values - critical_value * se_em
    upper = em_values + critical_value * se_em

    if family is not None and type == "response":
        eta = em_values
        mu = family.link.inverse(eta)
        link_derivative = family.link.deriv(mu)
        with np.errstate(divide="ignore", invalid="ignore"):
            response_derivative = 1.0 / link_derivative

        se_em = se_em * np.abs(response_derivative)
        lower_response = family.link.inverse(lower)
        upper_response = family.link.inverse(upper)
        em_values = mu
        lower = np.minimum(lower_response, upper_response)
        upper = np.maximum(lower_response, upper_response)

    result = EmmeanResult(
        emmean=em_values,
        se=se_em,
        df=df_resid,
        lower=lower,
        upper=upper,
        grid=result_grid,
        level=level,
    )

    return Emmeans(
        result=result,
        _L=L,
        _vcov=vcov,
        _beta=beta,
        _df=df_resid,
        _specs=[name for name in spec_names if name not in by_names],
        _levels=[levels[name] for name in spec_names if name not in by_names],
        _by=by_names,
    )
