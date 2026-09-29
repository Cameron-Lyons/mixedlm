from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

if TYPE_CHECKING:
    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult

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

    def pairs(
        self,
        adjust: str = "tukey",
        level: float = 0.95,
    ) -> ContrastResult:
        n_levels = len(self.result.emmean)
        if n_levels < 2:
            raise ValueError("Need at least 2 levels for pairwise comparisons")

        grid_labels = []
        for i in range(n_levels):
            parts = []
            for spec in self._specs:
                parts.append(str(self.result.grid.iloc[i][spec]))
            grid_labels.append(",".join(parts) if len(parts) > 1 else parts[0])

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
        n_levels = len(self.result.emmean)

        grid_labels = []
        for i in range(n_levels):
            parts = [str(self.result.grid.iloc[i][spec]) for spec in self._specs]
            grid_labels.append(",".join(parts) if len(parts) > 1 else parts[0])

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


def emmeans(
    model: LmerResult | GlmerResult,
    specs: str | list[str],
    _by: str | list[str] | None = None,
    at: dict[str, Any] | None = None,
    cov_reduce: Callable[[pd.Series], float] = np.mean,
    type: str = "response",
    level: float = 0.95,
) -> Emmeans:
    if type not in {"link", "response"}:
        raise ValueError("type must be 'link' or 'response'")

    if isinstance(specs, str):
        specs = [specs]

    frame = model.model_frame()
    terms = model.terms()
    beta = model.beta
    vcov = model.vcov()
    df_resid = float(model.df_residual())

    factor_vars: dict[str, list[Any]] = {}
    covariate_vars: dict[str, float] = {}

    for var in terms.fixed_variables:
        if var not in frame.columns:
            continue
        col = frame[var]
        dtype_str = str(col.dtype)
        is_string = "string" in dtype_str.lower() or "str" in dtype_str.lower()
        if col.dtype == object or col.dtype.name == "category" or is_string:
            if col.dtype.name == "category":
                levels = col.cat.categories.tolist()
            else:
                levels = sorted(col.dropna().unique().tolist())
            factor_vars[var] = levels
        else:
            covariate_vars[var] = float(cov_reduce(col))

    if at is not None:
        for var, val in at.items():
            if var in factor_vars:
                if isinstance(val, list):
                    factor_vars[var] = val
                else:
                    factor_vars[var] = [val]
            elif var in covariate_vars:
                covariate_vars[var] = float(val) if not isinstance(val, list) else float(val[0])

    for spec in specs:
        if spec not in factor_vars:
            raise ValueError(
                f"Variable '{spec}' must be a factor. Available factors: {list(factor_vars.keys())}"
            )

    spec_levels = [factor_vars[spec] for spec in specs]

    reference_levels = {**factor_vars, **{var: [value] for var, value in covariate_vars.items()}}
    L, result_grid = _marginal_mean_coefficients(model, specs, reference_levels)

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
        _specs=specs,
        _levels=spec_levels,
    )
