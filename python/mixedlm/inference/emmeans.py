from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import lru_cache
from math import prod
from numbers import Real
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from mixedlm.utils import _format_pvalue
from mixedlm.utils.dataframe import get_categories, get_column_numpy, is_categorical_or_string

if TYPE_CHECKING:
    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult

_MAX_CONTRAST_ELEMENTS = 1_000_000


def _contrast_moments(
    n_contrasts: int,
    coefficients: Callable[[slice], NDArray[np.floating]],
    beta: NDArray[np.floating],
    covariance: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate contrasts with bounded coefficient and projection buffers."""
    estimates = np.empty(n_contrasts, dtype=np.float64)
    variances = np.empty(n_contrasts, dtype=np.float64)
    chunk_size = max(1, _MAX_CONTRAST_ELEMENTS // max(1, len(beta)))
    for start in range(0, n_contrasts, chunk_size):
        rows = slice(start, min(start + chunk_size, n_contrasts))
        chunk = coefficients(rows)
        estimates[rows] = chunk @ beta
        variances[rows] = _rowwise_quadratic_form(chunk, covariance)
        del chunk
    return estimates, variances


_MAX_REFERENCE_GRID_ELEMENTS = 1_000_000


def _marginal_mean_coefficients(
    model: LmerResult | GlmerResult,
    result_names: list[str],
    levels: dict[str, list[Any]],
) -> tuple[NDArray[np.float64], pd.DataFrame]:
    """Average the fitted design over reference dimensions in bounded batches."""
    averaged_names = [name for name in levels if name not in result_names]
    all_names = result_names + averaged_names
    result_grid = pd.DataFrame(
        itertools.product(*(levels[name] for name in result_names)), columns=result_names
    )
    n_results = len(result_grid)
    n_averaged = prod(len(levels[name]) for name in averaged_names)
    n_beta = len(model.beta)
    coefficients = np.empty((n_results, n_beta), dtype=np.float64)
    if n_results == 0 or n_averaged == 0:
        coefficients.fill(np.nan)
        return coefficients, result_grid

    batch_rows = max(1, _MAX_REFERENCE_GRID_ELEMENTS // max(1, len(all_names), n_beta))
    combinations = itertools.product(*(levels[name] for name in all_names))

    if n_averaged <= batch_rows:
        # Keep each mean's reference rows together whenever they fit in a batch.
        results_per_batch = batch_rows // n_averaged
        for start in range(0, n_results, results_per_batch):
            count = min(results_per_batch, n_results - start)
            grid = pd.DataFrame(
                itertools.islice(combinations, count * n_averaged), columns=all_names
            )
            design = model._prediction_fixed_matrix(grid)
            coefficients[start : start + count] = design.reshape(count, n_averaged, n_beta).mean(
                axis=1
            )
            del grid, design
    else:
        # A single mean can span a large Cartesian product of nuisance levels.
        for result_index in range(n_results):
            total = np.zeros(n_beta, dtype=np.float64)
            for start in range(0, n_averaged, batch_rows):
                count = min(batch_rows, n_averaged - start)
                grid = pd.DataFrame(itertools.islice(combinations, count), columns=all_names)
                design = model._prediction_fixed_matrix(grid)
                total += design.sum(axis=0)
                del grid, design
            coefficients[result_index] = total / n_averaged

    return coefficients, result_grid


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
            max_width = max(len(str(col)), max((len(str(v)) for v in self.grid[col]), default=0))
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

    level: float = 0.95
    # Each family records its row bounds and, for pairwise differences, mean count.
    _families: tuple[tuple[int, int, int | None], ...] = field(default=(), repr=False)

    def confint(self, level: float | None = None, adjust: str | None = None) -> pd.DataFrame:
        """Return link-scale confidence intervals in the comparison row order.

        Defaults to the level and adjustment selected when creating the contrasts.
        Holm, FDR, and the current Dunnett approximation use Bonferroni intervals;
        the actual method is recorded in ``result.attrs["adjust"]``. Tukey requires
        pairwise differences, including custom rows with opposite coefficients.
        Grouped comparisons retain separate interval families. Quantiles are
        evaluated on demand.
        """
        confidence = _validate_contrast_level(self.level if level is None else level)
        requested = self.adjust if adjust is None else adjust
        if not isinstance(requested, str):
            raise TypeError("adjust must be a string naming an interval adjustment")
        requested = requested.strip().lower()
        if requested == "bh":
            requested = "fdr"
        if requested not in {"none", "tukey", "bonferroni", "holm", "fdr", "dunnett"}:
            raise ValueError(f"Unknown interval adjustment: {requested!r}")
        interval_adjust = requested if requested in {"none", "tukey"} else "bonferroni"
        families = self._families or ((0, len(self.estimate), None),)
        half_width = np.empty(len(self.estimate), dtype=np.float64)
        for start, stop, n_means in families:
            if start == stop:
                continue
            if interval_adjust == "tukey" and n_means is None:
                raise ValueError(
                    "Tukey intervals require pairwise differences; use "
                    "adjust='bonferroni' or 'none' for general custom contrasts."
                )
            critical = _contrast_critical_value(
                confidence, self.df, interval_adjust, stop - start, n_means
            )
            half_width[start:stop] = critical * self.se[start:stop]
        result = pd.DataFrame(
            {
                "contrast": self.contrast,
                "estimate": self.estimate,
                "SE": self.se,
                "df": self.df,
                "lower": self.estimate - half_width,
                "upper": self.estimate + half_width,
            },
            copy=True,
        )
        result.attrs.update(level=confidence, adjust=interval_adjust, requested_adjust=requested)
        return result

    def __str__(self) -> str:
        lines = []
        lines.append("Pairwise Comparisons")
        lines.append("")

        max_contrast_len = max((len(c) for c in self.contrast), default=0)
        max_contrast_len = max(max_contrast_len, 8)

        header = f"{'contrast':<{max_contrast_len}} {'estimate':>10} {'SE':>8}"
        ratio_label = "z.ratio" if np.isinf(self.df) else "t.ratio"
        header += f" {'df':>6} {ratio_label:>8} {'p.value':>10}"
        lines.append(header)

        for i in range(len(self.contrast)):
            p_str = _format_pvalue(self.p_value[i])
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
    _offset: NDArray[np.floating] | None = None

    def _grouped_contrasts(self, compute: Callable[[Emmeans], ContrastResult]) -> ContrastResult:
        """Compare means and adjust p-values independently within each by group."""
        results: list[ContrastResult] = []
        grids: list[pd.DataFrame] = []
        families: list[tuple[int, int, int | None]] = []
        row_offset = 0
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
            means = replace(
                self,
                result=subset,
                _L=self._L[indices],
                _by=[],
                _offset=None if self._offset is None else self._offset[indices],
            )
            result = compute(means)
            values = {name: grid[name].iloc[0] for name in self._by}
            description = ", ".join(f"{name}={value}" for name, value in values.items())
            labels.extend(f"{label} | {description}" for label in result.contrast)
            grids.append(pd.DataFrame([values] * len(result.contrast), columns=self._by))
            families.extend(
                (start + row_offset, stop + row_offset, n_means)
                for start, stop, n_means in result._families
            )
            row_offset += len(result.estimate)
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
            level=results[0].level,
            _families=tuple(families),
        )

    def pairs(
        self,
        adjust: str = "tukey",
        level: float = 0.95,
    ) -> ContrastResult:
        level = _validate_contrast_level(level)
        adjust = _normalize_adjustment(adjust)
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
        estimates, var_contrast = _contrast_moments(
            len(left_indices),
            lambda rows: self._L[left_indices[rows]] - self._L[right_indices[rows]],
            self._beta,
            self._vcov,
        )
        if self._offset is not None:
            estimates += self._offset[left_indices] - self._offset[right_indices]
        se_contrast = np.sqrt(np.maximum(var_contrast, 0))

        t_ratio = estimates / se_contrast

        raw_p = 2 * stats.t.sf(np.abs(t_ratio), self._df)
        p_adjusted = _adjust_pvalues(raw_p, adjust, n_levels, self._df, t_ratio)

        return ContrastResult(
            contrast=contrast_labels,
            estimate=estimates,
            se=se_contrast,
            df=self._df,
            t_ratio=t_ratio,
            p_value=p_adjusted,
            adjust=adjust,
            level=level,
            _families=((0, len(contrast_labels), n_levels),),
        )

    def contrast(
        self,
        method: str | ArrayLike = "pairwise",
        adjust: str | None = None,
        level: float = 0.95,
    ) -> ContrastResult:
        """Compute contrasts with Tukey as the default for pairwise comparisons.

        Other contrast methods default to no adjustment. Pass ``adjust="none"``
        explicitly for unadjusted pairwise tests; ``None`` selects the default.
        """
        if adjust is None:
            adjust = "tukey" if isinstance(method, str) and method == "pairwise" else "none"
        if isinstance(method, str):
            if method == "pairwise":
                return self.pairs(adjust=adjust, level=level)
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
        level = _validate_contrast_level(level)
        adjust = _normalize_adjustment(adjust)
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
        estimates, var_contrast = _contrast_moments(
            len(treatment_indices),
            lambda rows: self._L[treatment_indices[rows]] - self._L[ctrl_idx],
            self._beta,
            self._vcov,
        )
        if self._offset is not None:
            estimates += self._offset[treatment_indices] - self._offset[ctrl_idx]
        se_contrast = np.sqrt(np.maximum(var_contrast, 0))
        t_ratio = estimates / se_contrast
        raw_p = 2 * stats.t.sf(np.abs(t_ratio), self._df)
        p_adjusted = _adjust_pvalues(raw_p, adjust, n_levels, self._df, t_ratio)

        return ContrastResult(
            contrast=contrast_labels,
            estimate=estimates,
            se=se_contrast,
            df=self._df,
            t_ratio=t_ratio,
            p_value=p_adjusted,
            adjust=adjust,
            level=level,
            _families=((0, len(contrast_labels), n_levels),),
        )

    def _custom_contrast(
        self,
        C: ArrayLike,
        adjust: str = "none",
        level: float = 0.95,
    ) -> ContrastResult:
        level = _validate_contrast_level(level)
        adjust = _normalize_adjustment(adjust)
        if self._by:
            return self._grouped_contrasts(
                lambda means: means._custom_contrast(C, adjust=adjust, level=level)
            )
        n_levels = len(self.result.emmean)
        C, pairwise_means = _validate_custom_contrasts(C, n_levels)
        if adjust == "tukey" and pairwise_means is None and len(C):
            raise ValueError(
                "Tukey adjustment requires pairwise differences; use "
                "adjust='bonferroni', 'holm', 'fdr', or 'none' for general custom contrasts."
            )
        n_contrasts = C.shape[0]
        estimates, var_contrast = _contrast_moments(
            n_contrasts, lambda rows: C[rows] @ self._L, self._beta, self._vcov
        )
        if self._offset is not None:
            estimates += C @ self._offset
        se_contrast = np.sqrt(np.maximum(var_contrast, 0))
        t_ratio = estimates / se_contrast
        raw_p = 2 * stats.t.sf(np.abs(t_ratio), self._df)
        p_adjusted = _adjust_pvalues(
            raw_p, adjust, n_levels if adjust == "tukey" else n_contrasts, self._df, t_ratio
        )

        contrast_labels = [f"C{i + 1}" for i in range(n_contrasts)]

        return ContrastResult(
            contrast=contrast_labels,
            estimate=estimates,
            se=se_contrast,
            df=self._df,
            t_ratio=t_ratio,
            p_value=p_adjusted,
            adjust=adjust,
            level=level,
            _families=((0, n_contrasts, pairwise_means),),
        )

    def __str__(self) -> str:
        return str(self.result)

    def __repr__(self) -> str:
        return f"Emmeans(specs={self._specs}, n={len(self.result.emmean)})"


_ADJUSTMENT_METHODS = ("none", "bonferroni", "holm", "fdr", "tukey", "dunnett")


def _normalize_adjustment(method: str) -> str:
    if not isinstance(method, str):
        raise TypeError("adjust must be a string naming a p-value adjustment")
    normalized = method.strip().lower()
    if normalized == "bh":
        normalized = "fdr"
    if normalized not in _ADJUSTMENT_METHODS:
        choices = ", ".join(_ADJUSTMENT_METHODS)
        raise ValueError(f"Unknown p-value adjustment: {method!r}. Choose from {choices}, or BH.")
    return normalized


def _validate_custom_contrasts(C: ArrayLike, n_means: int) -> tuple[NDArray, int | None]:
    """Validate real coefficient rows and identify scaled pairwise differences."""
    if np.ma.is_masked(C):
        raise ValueError("Custom contrast coefficients must not contain masked values")
    try:
        coefficients = np.asarray(C)
    except (TypeError, ValueError) as exc:
        raise ValueError("Custom contrast coefficients must form a rectangular 2-D matrix") from exc
    if coefficients.ndim != 2:
        raise ValueError(
            "Custom contrast coefficients must be a 2-D matrix; use [[...]] for a single contrast"
        )
    if coefficients.shape[1] != n_means:
        raise ValueError(
            f"Custom contrast matrix has {coefficients.shape[1]} columns; "
            f"expected {n_means}, one per marginal mean"
        )
    if np.iscomplexobj(coefficients) or coefficients.dtype.kind in "mMV":
        raise TypeError("Custom contrast coefficients must be real numeric values")
    if coefficients.dtype.kind not in "biuf":
        try:
            coefficients = coefficients.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise TypeError("Custom contrast coefficients must be real numeric values") from exc

    pairwise = n_means >= 2 and coefficients.dtype.kind != "b"
    batch_rows = max(1, _MAX_CONTRAST_ELEMENTS // max(1, n_means))
    for start in range(0, len(coefficients), batch_rows):
        chunk = coefficients[start : start + batch_rows]
        if not np.isfinite(chunk).all():
            raise ValueError("Custom contrast coefficients must contain only finite values")
        if pairwise:
            minimum = chunk.min(axis=1)
            maximum = chunk.max(axis=1)
            pairwise = bool(np.all((minimum < 0) & (maximum > 0) & (minimum == -maximum)))
            if pairwise:
                pairwise = bool(np.all(np.count_nonzero(chunk, axis=1) == 2))
    return coefficients, n_means if pairwise else None


def _validate_contrast_level(level: float) -> float:
    if isinstance(level, bool | np.bool_) or not isinstance(level, Real):
        raise TypeError("level must be a finite number strictly between 0 and 1")
    value = float(level)
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError("level must be a finite number strictly between 0 and 1")
    return value


@lru_cache(maxsize=128)
def _contrast_critical_value(
    level: float, df: float, adjust: str, n_comparisons: int, n_means: int | None
) -> float:
    """Share scalar quantiles across interval families without retaining arrays."""
    alpha = 1.0 - level
    if adjust == "tukey" and n_means != 2:
        return float(stats.studentized_range.isf(alpha, n_means, df) / np.sqrt(2.0))
    tail = alpha / (2 * n_comparisons) if adjust == "bonferroni" else alpha / 2
    return float(stats.norm.isf(tail) if np.isinf(df) else stats.t.isf(tail, df))


def _adjust_pvalues(
    p: NDArray[np.floating],
    method: str,
    n_groups: int,
    df: float,
    t_ratio: NDArray[np.floating] | None = None,
) -> NDArray[np.floating]:
    method = _normalize_adjustment(method)
    if method == "none":
        return p
    elif method == "bonferroni":
        return np.minimum(p * len(p), 1.0)
    elif method in ("holm", "fdr"):
        n = len(p)
        sorted_idx = np.argsort(p)
        ranked = np.asarray(p[sorted_idx], dtype=np.float64)
        if method == "holm":
            ranked *= np.arange(n, 0, -1)
            np.maximum.accumulate(ranked, out=ranked)
        else:
            ranked *= n
            ranked /= np.arange(1, n + 1)
            # NaNs sort last: ignore them when accumulating backwards while
            # retaining the full comparison count and their undefined outputs.
            np.fmin.accumulate(ranked[::-1], out=ranked[::-1])
        np.minimum(ranked, 1.0, out=ranked)
        adjusted = np.empty(n, dtype=np.float64)
        adjusted[sorted_idx] = ranked
        return adjusted
    elif method == "tukey":
        if t_ratio is None:
            raise ValueError("t_ratio is required for Tukey adjustment")
        q = np.abs(t_ratio) * np.sqrt(2)
        return stats.studentized_range.sf(q, n_groups, df)

    # The remaining supported method, dunnett, retains its Bonferroni approximation.
    return np.minimum(p * len(p), 1.0)


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


def _reference_offsets(
    model: LmerResult | GlmerResult,
    offset: ArrayLike | None,
    n_means: int,
) -> NDArray[np.floating]:
    """Resolve known offsets on the link scale, in result-grid row order."""
    if offset is None:
        offset = np.mean(model.matrices.offset)
    if np.iscomplexobj(offset):
        raise ValueError("offset must contain real numeric values")
    try:
        values = np.asarray(offset, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("offset must be a finite numeric scalar or 1-D sequence") from None
    if values.ndim == 0:
        values = np.full(n_means, values.item(), dtype=np.float64)
    elif values.ndim != 1 or len(values) != n_means:
        raise ValueError(f"offset must be a scalar or a 1-D sequence with {n_means} values")
    if not np.all(np.isfinite(values)):
        raise ValueError("offset must contain only finite values")
    return values.copy()


def emmeans(
    model: LmerResult | GlmerResult,
    specs: str | list[str],
    by: str | list[str] | None = None,
    at: dict[str, Any] | None = None,
    cov_reduce: Callable[[pd.Series], float] = np.mean,
    type: str = "response",
    level: float = 0.95,
    *,
    offset: ArrayLike | None = None,
    _by: str | list[str] | None = None,
) -> Emmeans:
    """Estimate means over a reference grid, optionally comparing within by groups.

    Numeric predictors use ``cov_reduce`` unless ``at`` supplies reference values.
    Predictors in ``specs`` or ``by`` identify result rows; all other reference
    dimensions are averaged with equal weights. Contrasts compare means within
    each ``by`` group and adjust that group's p-values separately.

    ``offset=None`` uses the unweighted mean of the fitted link-scale offsets
    after missing-value omission. A finite scalar or one value per result-grid
    row overrides that reference offset. Use ``offset=0`` for per-unit rates
    in a count model fitted with a log-exposure offset. Offsets also enter
    contrasts but contribute no additional coefficient uncertainty.
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
    family = getattr(model, "family", None)
    df = np.inf if family is not None else float(model.df_residual())
    L, result_grid = _marginal_mean_coefficients(model, result_names, levels)
    offsets = _reference_offsets(model, offset, len(result_grid))
    vcov = model.vcov()

    em_values = L @ beta + offsets
    var_em = _rowwise_quadratic_form(L, vcov)
    se_em = np.sqrt(np.maximum(var_em, 0))

    alpha = 1 - level

    if family is not None:
        critical_value = stats.norm.ppf(1 - alpha / 2)
    else:
        critical_value = stats.t.ppf(1 - alpha / 2, df)

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
        df=df,
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
        _df=df,
        _specs=[name for name in spec_names if name not in by_names],
        _levels=[levels[name] for name in spec_names if name not in by_names],
        _by=by_names,
        _offset=offsets,
    )
