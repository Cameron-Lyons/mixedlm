from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

if TYPE_CHECKING:
    from mixedlm.inference.bootstrap import BootstrapResult, NlmerBootstrapResult


@dataclass(frozen=True)
class _SampleBlock:
    component: str
    names: list[str]
    samples: NDArray[np.floating]
    original: NDArray[np.floating]


_METHOD_ALIASES = {
    "percentile": "percentile",
    "perc": "percentile",
    "basic": "basic",
    "normal": "normal",
    "norm": "normal",
}
_COMPONENTS = {"fixed", "theta", "sigma", "all"}
_RESULT_COLUMNS = [
    "parameter",
    "component",
    "method",
    "estimate",
    "mean",
    "bias",
    "std.error",
    "conf.low",
    "conf.high",
    "n.success",
]


def _normalize_methods(method: str | Sequence[str]) -> list[str]:
    if isinstance(method, str):
        requested = [method]
    else:
        try:
            requested = list(method)
        except TypeError as exc:
            raise TypeError("method must be an interval name or sequence of names") from exc
    if not requested:
        raise ValueError("method must contain at least one interval type")

    normalized: list[str] = []
    for value in requested:
        if not isinstance(value, str):
            raise TypeError("method must contain interval names")
        key = value.lower()
        if key not in _METHOD_ALIASES:
            choices = ", ".join(sorted(_METHOD_ALIASES))
            raise ValueError(f"Unknown method '{value}'. Choose from: {choices}")
        canonical = _METHOD_ALIASES[key]
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized


def _as_matrix(samples: Any, name: str) -> NDArray[np.float64]:
    array = np.asarray(samples, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"{name} must be a one- or two-dimensional sample array")
    return array


def _sample_blocks(
    result: BootstrapResult | NlmerBootstrapResult,
) -> dict[str, _SampleBlock]:
    blocks: dict[str, _SampleBlock] = {}

    if hasattr(result, "beta_samples"):
        linear_result = cast("BootstrapResult", result)
        fixed_samples = _as_matrix(linear_result.beta_samples, "beta_samples")
        fixed_names = list(linear_result.fixed_names)
        fixed_original = np.asarray(linear_result.original_beta, dtype=np.float64)
    elif hasattr(result, "phi_samples"):
        nonlinear_result = cast("NlmerBootstrapResult", result)
        fixed_samples = _as_matrix(nonlinear_result.phi_samples, "phi_samples")
        fixed_names = list(nonlinear_result.param_names)
        fixed_original = np.asarray(nonlinear_result.original_phi, dtype=np.float64)
    else:
        raise TypeError("result must be returned by bootMer() or bootstrap_nlmer()")

    blocks["fixed"] = _SampleBlock(
        component="fixed",
        names=fixed_names,
        samples=fixed_samples,
        original=fixed_original,
    )

    theta_samples = _as_matrix(result.theta_samples, "theta_samples")
    theta_original = np.asarray(result.original_theta, dtype=np.float64)
    blocks["theta"] = _SampleBlock(
        component="theta",
        names=[f"theta[{index}]" for index in range(1, theta_samples.shape[1] + 1)],
        samples=theta_samples,
        original=theta_original,
    )

    sigma_samples = getattr(result, "sigma_samples", None)
    sigma_original = getattr(result, "original_sigma", None)
    if sigma_samples is not None and sigma_original is not None:
        blocks["sigma"] = _SampleBlock(
            component="sigma",
            names=["sigma"],
            samples=_as_matrix(sigma_samples, "sigma_samples"),
            original=np.array([sigma_original], dtype=np.float64),
        )

    for block in blocks.values():
        if block.samples.shape[0] != int(result.n_boot):
            raise ValueError(f"{block.component} sample rows do not match n_boot={result.n_boot}")
        if block.samples.shape[1] != len(block.names):
            raise ValueError(f"{block.component} sample columns do not match their parameter names")
        if block.original.shape != (len(block.names),):
            raise ValueError(
                f"{block.component} original estimates do not match their parameter names"
            )
        if not np.all(np.isfinite(block.original)):
            raise ValueError(f"{block.component} original estimates must be finite")
    return blocks


def _selected_blocks(blocks: dict[str, _SampleBlock], component: str) -> list[_SampleBlock]:
    if component not in _COMPONENTS:
        choices = ", ".join(sorted(_COMPONENTS))
        raise ValueError(f"Unknown component '{component}'. Choose from: {choices}")
    if component == "all":
        return [blocks[name] for name in ("fixed", "theta", "sigma") if name in blocks]
    if component not in blocks:
        raise ValueError(f"Component '{component}' is not available for this bootstrap result")
    return [blocks[component]]


def _interval_rows(
    block: _SampleBlock,
    methods: list[str],
    level: float,
) -> list[dict[str, Any]]:
    alpha = 1.0 - level
    quantiles = np.array([alpha / 2.0, 1.0 - alpha / 2.0])
    z_value = float(stats.norm.ppf(1.0 - alpha / 2.0))
    rows: list[dict[str, Any]] = []

    for index, name in enumerate(block.names):
        estimate = float(block.original[index])
        samples = block.samples[:, index]
        samples = samples[np.isfinite(samples)]
        n_success = len(samples)

        if n_success:
            sample_mean = float(np.mean(samples))
            bias = sample_mean - estimate
            standard_error = float(np.std(samples, ddof=1)) if n_success > 1 else np.nan
            lower_quantile, upper_quantile = np.quantile(samples, quantiles)
        else:
            sample_mean = np.nan
            bias = np.nan
            standard_error = np.nan
            lower_quantile = np.nan
            upper_quantile = np.nan

        for interval_method in methods:
            if interval_method == "percentile":
                lower = lower_quantile
                upper = upper_quantile
            elif interval_method == "basic":
                lower = 2.0 * estimate - upper_quantile
                upper = 2.0 * estimate - lower_quantile
            else:
                center = estimate - bias
                lower = center - z_value * standard_error
                upper = center + z_value * standard_error

            rows.append(
                {
                    "parameter": name,
                    "component": block.component,
                    "method": interval_method,
                    "estimate": estimate,
                    "mean": sample_mean,
                    "bias": bias,
                    "std.error": standard_error,
                    "conf.low": float(lower),
                    "conf.high": float(upper),
                    "n.success": n_success,
                }
            )
    return rows


def bootCI(
    result: BootstrapResult | NlmerBootstrapResult,
    *,
    level: float = 0.95,
    method: str | Sequence[str] = "percentile",
    component: str = "fixed",
    parameters: str | Sequence[str] | None = None,
) -> pd.DataFrame:
    """Summarize bootstrap confidence intervals in a tidy data frame.

    Parameters
    ----------
    result : BootstrapResult or NlmerBootstrapResult
        Result returned by :func:`bootMer` or :func:`bootstrap_nlmer`.
    level : float, default 0.95
        Confidence level.
    method : str or sequence of str, default "percentile"
        One or more of ``"percentile"``, ``"basic"``, or ``"normal"``.
        The short aliases ``"perc"`` and ``"norm"`` are accepted.
    component : {"fixed", "theta", "sigma", "all"}, default "fixed"
        Parameter family to include. Residual scale is unavailable for GLMMs.
    parameters : str or sequence of str, optional
        Restrict output to named parameters after selecting the component.

    Returns
    -------
    pandas.DataFrame
        One row per parameter and interval method, including the original
        estimate, bootstrap mean, bias, sample standard error, interval bounds,
        and successful replicate count.
    """
    try:
        level_value = float(level)
    except (TypeError, ValueError) as exc:
        raise TypeError("level must be a number between 0 and 1") from exc
    if not 0.0 < level_value < 1.0:
        raise ValueError("level must be between 0 and 1")
    if not isinstance(component, str):
        raise TypeError("component must be a string")

    methods = _normalize_methods(method)
    blocks = _selected_blocks(_sample_blocks(result), component.lower())
    rows = [row for block in blocks for row in _interval_rows(block, methods, level_value)]

    if parameters is not None:
        if isinstance(parameters, str):
            selected = [parameters]
        else:
            try:
                selected = list(parameters)
            except TypeError as exc:
                raise TypeError("parameters must be a name or sequence of names") from exc
        if not selected:
            raise ValueError("parameters cannot be empty")
        if any(not isinstance(name, str) or not name for name in selected):
            raise TypeError("parameters must contain non-empty names")
        available = {row["parameter"] for row in rows}
        missing = [name for name in selected if name not in available]
        if missing:
            raise ValueError(
                f"Unknown parameter(s): {', '.join(missing)}. "
                f"Available parameters: {', '.join(sorted(available))}"
            )
        selected_set = set(selected)
        rows = [row for row in rows if row["parameter"] in selected_set]

    frame = pd.DataFrame(rows, columns=_RESULT_COLUMNS)
    frame.attrs.update(
        {
            "level": level_value,
            "component": component.lower(),
            "methods": methods,
            "n_boot": int(result.n_boot),
            "n_failed": int(result.n_failed),
        }
    )
    return frame
