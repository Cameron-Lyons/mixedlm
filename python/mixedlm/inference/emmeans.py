from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

if TYPE_CHECKING:
    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult


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

        n_pairs = n_levels * (n_levels - 1) // 2
        C = np.zeros((n_pairs, n_levels), dtype=np.float64)
        contrast_labels: list[str] = []

        grid_labels = []
        for i in range(n_levels):
            parts = []
            for spec in self._specs:
                parts.append(str(self.result.grid.iloc[i][spec]))
            grid_labels.append(",".join(parts) if len(parts) > 1 else parts[0])

        k = 0
        for i in range(n_levels):
            for j in range(i + 1, n_levels):
                C[k, i] = 1
                C[k, j] = -1
                contrast_labels.append(f"{grid_labels[i]} - {grid_labels[j]}")
                k += 1

        L_contrast = C @ self._L

        estimates = L_contrast @ self._beta
        var_contrast = np.diag(L_contrast @ self._vcov @ L_contrast.T)
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
        n_contrasts = n_levels - 1

        C = np.zeros((n_contrasts, n_levels), dtype=np.float64)
        contrast_labels: list[str] = []

        grid_labels = []
        for i in range(n_levels):
            parts = [str(self.result.grid.iloc[i][spec]) for spec in self._specs]
            grid_labels.append(",".join(parts) if len(parts) > 1 else parts[0])

        k = 0
        for i in range(n_levels):
            if i != ctrl_idx:
                C[k, i] = 1
                C[k, ctrl_idx] = -1
                contrast_labels.append(f"{grid_labels[i]} - {grid_labels[ctrl_idx]}")
                k += 1

        L_contrast = C @ self._L
        estimates = L_contrast @ self._beta
        var_contrast = np.diag(L_contrast @ self._vcov @ L_contrast.T)
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
        var_contrast = np.diag(L_contrast @ self._vcov @ L_contrast.T)
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
    cov_reduce: Callable[[pd.Series], float] = np.mean,  # type: ignore[type-arg]
    type: str = "response",
    level: float = 0.95,
) -> Emmeans:
    from mixedlm.matrices.design import build_fixed_matrix

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

    other_factors = {k: v for k, v in factor_vars.items() if k not in specs}

    if other_factors:
        all_vars = specs + list(other_factors.keys())
        all_levels = spec_levels + list(other_factors.values())
    else:
        all_vars = specs
        all_levels = spec_levels

    grid_data: dict[str, list[Any]] = {var: [] for var in all_vars}
    for cov_var in covariate_vars:
        grid_data[cov_var] = []

    combinations = list(itertools.product(*all_levels))
    for combo in combinations:
        for i, var in enumerate(all_vars):
            grid_data[var].append(combo[i])
        for cov_var, cov_val in covariate_vars.items():
            grid_data[cov_var].append(cov_val)

    grid = pd.DataFrame(grid_data)

    X_grid, _ = build_fixed_matrix(model.formula, grid)

    spec_combinations = list(itertools.product(*spec_levels))
    n_emmeans = len(spec_combinations)
    n_beta = len(beta)

    L = np.zeros((n_emmeans, n_beta), dtype=np.float64)

    for i, spec_combo in enumerate(spec_combinations):
        mask = np.ones(len(grid), dtype=bool)
        for j, spec in enumerate(specs):
            mask &= grid[spec] == spec_combo[j]

        X_subset = X_grid[mask]
        L[i] = X_subset.mean(axis=0)

    em_values = L @ beta
    var_em = np.diag(L @ vcov @ L.T)
    se_em = np.sqrt(np.maximum(var_em, 0))

    is_glmm = hasattr(model, "family") and model.family is not None

    if is_glmm and type == "response":
        eta = em_values
        mu = model.family.link.inverse(eta)  # type: ignore[union-attr]
        deriv = model.family.link.deriv(mu)  # type: ignore[union-attr]
        se_response = se_em * np.abs(deriv)
        em_values = mu
        se_em = se_response
        t_crit = stats.norm.ppf(1 - (1 - level) / 2)
    else:
        t_crit = stats.t.ppf(1 - (1 - level) / 2, df_resid)

    lower = em_values - t_crit * se_em
    upper = em_values + t_crit * se_em

    result_grid_data: dict[str, list[Any]] = {spec: [] for spec in specs}
    for spec_combo in spec_combinations:
        for j, spec in enumerate(specs):
            result_grid_data[spec].append(spec_combo[j])
    result_grid = pd.DataFrame(result_grid_data)

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
