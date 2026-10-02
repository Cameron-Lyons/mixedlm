from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from mixedlm.estimation.validation import validate_finite_real
from mixedlm.utils.names import _check_unique_coefficient_names
from mixedlm.utils.random import validate_simulation_count

if TYPE_CHECKING:
    import pandas as pd

    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult

_DEFAULT_NSIM = 1000
_DEFAULT_NSIM_CURVE = 500
_DEFAULT_ALPHA = 0.05
_CI_ALPHA = 0.05
_POWER_THRESHOLD = 0.8
_VERBOSE_INTERVAL = 100
_MIN_GROUPS = 5
_DEFAULT_EFFECT_SIZE_VALUES = [0.5, 0.75, 1.0, 1.25, 1.5]
_SEED_OFFSET_MULTIPLIER = 1000
_FIGURE_SIZE = (8, 5)


def _new_group_labels(existing: list[Any], count: int) -> list[Any]:
    existing_set = set(existing)
    labels: list[Any] = []

    if existing and all(
        isinstance(value, Real) and not isinstance(value, bool) for value in existing
    ):
        largest = max(existing)
        candidate = int(largest) if isinstance(largest, Integral) else float(largest)
        if isinstance(candidate, int) and candidate >= np.iinfo(np.int64).max:
            candidate = -1
        while len(labels) < count:
            successor = candidate + 1
            if successor == candidate and isinstance(candidate, float):
                with np.errstate(over="ignore"):
                    successor = float(np.nextafter(candidate, np.inf))
            # Use unused small numbers when the existing label has no finite
            # successor. Python integers also avoid NumPy integer wraparound.
            candidate = successor if isinstance(successor, int) or np.isfinite(successor) else 0.0
            if candidate not in existing_set:
                labels.append(candidate)
                existing_set.add(candidate)
        return labels

    suffix = 1
    while len(labels) < count:
        string_label = f"new_group_{suffix}"
        suffix += 1
        if string_label not in existing_set:
            labels.append(string_label)
            existing_set.add(string_label)
    return labels


@dataclass
class PowerResult:
    power: float
    ci_lower: float
    ci_upper: float
    n_successes: int
    n_simulations: int
    effect_size: float | None
    n_obs: int
    n_groups: int | None
    n_failed: int = 0

    def __str__(self) -> str:
        lines = [
            "Power analysis by simulation",
            "",
            f"Power: {self.power:.3f} (95% CI: [{self.ci_lower:.3f}, {self.ci_upper:.3f}])",
            f"Simulations: {self.n_simulations} ({self.n_successes} significant)",
            f"Observations: {self.n_obs}",
        ]
        if self.n_groups is not None:
            lines.append(f"Groups: {self.n_groups}")
        if self.effect_size is not None:
            lines.append(f"Effect size: {self.effect_size:.4f}")
        if self.n_failed:
            lines.append(f"Failed simulations: {self.n_failed}")
        return "\n".join(lines)


@dataclass
class PowerCurveResult:
    values: list[int | float]
    powers: list[float]
    ci_lowers: list[float]
    ci_uppers: list[float]
    along: str
    n_simulations: int
    results: tuple[PowerResult, ...] = field(default_factory=tuple)

    def __str__(self) -> str:
        lines = [f"Power curve along '{self.along}':", ""]
        lines.append(f"{'Value':>10} {'Power':>10} {'95% CI':>20}")
        lines.append("-" * 42)
        for v, p, lo, hi in zip(
            self.values, self.powers, self.ci_lowers, self.ci_uppers, strict=False
        ):
            lines.append(f"{v:>10} {p:>10.3f} [{lo:.3f}, {hi:.3f}]")
        return "\n".join(lines)

    def plot(self, ax=None, show_ci: bool = True):
        try:
            import matplotlib.pyplot as plt
        except ImportError as err:
            raise ImportError("matplotlib required for plotting") from err

        if ax is None:
            fig, ax = plt.subplots(figsize=_FIGURE_SIZE)
        else:
            fig = ax.get_figure()

        ax.plot(self.values, self.powers, "o-", linewidth=2, markersize=8)

        if show_ci:
            ax.fill_between(
                self.values,
                self.ci_lowers,
                self.ci_uppers,
                alpha=0.2,
            )

        ax.axhline(_POWER_THRESHOLD, color="red", linestyle="--", alpha=0.7, label="80% power")
        ax.set_xlabel(self.along)
        ax.set_ylabel("Power")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig


def _default_test(
    result: LmerResult | GlmerResult,
    param: str,
    alpha: float = _DEFAULT_ALPHA,
) -> bool:
    vcov = result.vcov()
    beta = result.beta
    validate_finite_real("fixed-effect covariance", vcov, (len(beta), len(beta)))

    param_names = result.matrices.fixed_names
    if param not in param_names:
        raise ValueError(f"Parameter '{param}' not found. Available: {param_names}")

    _check_unique_coefficient_names(
        param_names,
        [param],
        alternative="Use a callable test that selects coefficients by position.",
    )
    idx = param_names.index(param)
    if vcov[idx, idx] <= 0:
        raise ValueError(f"Parameter {param!r} has nonpositive sampling variance")
    se = np.sqrt(vcov[idx, idx])
    z_val = beta[idx] / se

    p_val = 2 * stats.norm.sf(np.abs(z_val))
    return bool(p_val < alpha)


def _binomial_score_interval(
    n_successes: int,
    n_trials: int,
    alpha: float,
) -> tuple[float, float]:
    """Compute a Wilson score interval for a binomial proportion."""
    proportion = n_successes / n_trials
    z_critical = float(stats.norm.isf(float(alpha) / 2.0))
    z_squared = z_critical**2
    denominator = 1.0 + z_squared / n_trials
    center = (proportion + z_squared / (2.0 * n_trials)) / denominator
    half_width = (
        z_critical
        * np.sqrt(proportion * (1.0 - proportion) / n_trials + z_squared / (4.0 * n_trials**2))
        / denominator
    )
    lower = 0.0 if n_successes == 0 else max(0.0, center - half_width)
    upper = 1.0 if n_successes == n_trials else min(1.0, center + half_width)
    return lower, upper


def _simulate_with_isolated_seed(
    model: LmerResult | GlmerResult,
    seed: int,
) -> NDArray[np.floating]:
    """Simulate reproducibly without changing NumPy's process-wide RNG state."""
    random_state = np.random.get_state()
    try:
        return model.simulate(nsim=1, seed=seed, use_re=True)
    finally:
        np.random.set_state(random_state)


def powerSim(
    model: LmerResult | GlmerResult,
    test: Callable[[LmerResult | GlmerResult], bool] | str | None = None,
    nsim: int = _DEFAULT_NSIM,
    alpha: float = _DEFAULT_ALPHA,
    seed: int | None = None,
    verbose: bool = False,
) -> PowerResult:
    """Estimate power via simulation.

    Simulates new data from the fitted model, refits, and counts
    significant results.

    Parameters
    ----------
    model : LmerResult or GlmerResult
        A fitted mixed model.
    test : callable or str, optional
        Either a callable that takes a fitted model and returns True
        if the test is significant, or a parameter name to test.
        If None, tests the first non-intercept coefficient, or the intercept
        in an intercept-only model. Callables may test models without fixed effects.
    nsim : int, default 1000
        Number of simulations.
    alpha : float, default 0.05
        Significance level.
    seed : int, optional
        Random seed for reproducibility. The process-wide NumPy random state
        is preserved.
    verbose : bool, default False
        Print progress information.

    Returns
    -------
    PowerResult
        Object containing the power estimate and a 95% Wilson score interval.
        Unconverged or invalid refits are excluded and counted in ``n_failed``;
        ``n_simulations`` is the number of valid completed simulations.

    Examples
    --------
    >>> result = lmer("y ~ x + (1|group)", data)
    >>> power = powerSim(result, test="x", nsim=500)
    >>> print(power)
    """
    _validate_power_options(nsim, alpha)
    param_names = model.matrices.fixed_names

    rng = np.random.default_rng(seed)

    def make_test_func(param: str, a: float) -> Callable[[LmerResult | GlmerResult], bool]:
        def test_fn(m: LmerResult | GlmerResult) -> bool:
            return _default_test(m, param, a)

        return test_fn

    test_param: str | None = None
    if test is None:
        test_param = _default_parameter(model)
        test_func = make_test_func(test_param, alpha)
    elif isinstance(test, str):
        if test not in param_names:
            raise ValueError(f"Parameter '{test}' not found. Available: {param_names}")
        test_param = test
        test_func = make_test_func(test_param, alpha)
    elif callable(test):
        test_func = test
    else:
        raise TypeError("test must be a parameter name, callable, or None")

    n_groups = None
    if hasattr(model, "ngrps"):
        grps = model.ngrps()
        if grps:
            n_groups = list(grps.values())[0]

    effect_size = None
    if test_param is not None:
        _check_unique_coefficient_names(
            param_names,
            [test_param],
            alternative="Use a callable test that selects coefficients by position.",
        )
        idx = param_names.index(test_param)
        effect_size = float(model.beta[idx])

    n_successes = 0
    n_completed = 0
    first_error: Exception | None = None

    for i in range(nsim):
        if verbose and (i + 1) % _VERBOSE_INTERVAL == 0:
            print(f"Simulation {i + 1}/{nsim}")

        try:
            simulation_seed = int(rng.integers(0, 2**32, dtype=np.uint64))
            y_sim = _simulate_with_isolated_seed(model, simulation_seed)
            fit_sim = model.refit(y_sim)
            _validate_power_refit(fit_sim, model)
            significant = test_func(fit_sim)
            if not isinstance(significant, bool | np.bool_):
                raise TypeError("test must return a boolean significance decision")
            if significant:
                n_successes += 1
            n_completed += 1
        except Exception as exc:
            if first_error is None:
                first_error = exc
            continue

    if n_completed == 0:
        error_detail = f" First error: {first_error}" if first_error is not None else ""
        warnings.warn(
            f"All {nsim} power simulations failed.{error_detail}",
            RuntimeWarning,
            stacklevel=2,
        )
        return PowerResult(
            power=np.nan,
            ci_lower=np.nan,
            ci_upper=np.nan,
            n_successes=0,
            n_simulations=0,
            effect_size=effect_size,
            n_obs=model.matrices.n_obs,
            n_groups=n_groups,
            n_failed=nsim,
        )

    power = n_successes / n_completed
    if n_completed < nsim:
        error_detail = f" First error: {first_error}" if first_error is not None else ""
        warnings.warn(
            f"{nsim - n_completed} of {nsim} power simulations failed.{error_detail}",
            RuntimeWarning,
            stacklevel=2,
        )

    ci_lower, ci_upper = _binomial_score_interval(n_successes, n_completed, _CI_ALPHA)

    return PowerResult(
        power=power,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_successes=n_successes,
        n_simulations=n_completed,
        effect_size=effect_size,
        n_obs=model.matrices.n_obs,
        n_groups=n_groups,
        n_failed=nsim - n_completed,
    )


def _validate_power_options(nsim: int, alpha: float) -> None:
    validate_simulation_count(nsim, "nsim")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, Real)
        or not np.isfinite(alpha)
        or not 0.0 < alpha < 1.0
    ):
        raise ValueError("alpha must be strictly between 0 and 1")


def _default_parameter(model: LmerResult | GlmerResult) -> str:
    names = model.matrices.fixed_names
    if not names:
        raise ValueError("model has no fixed effects to test; supply a callable test")
    return next((name for name in names if name != "(Intercept)"), names[0])


def _validate_power_refit(
    fitted: LmerResult | GlmerResult, original: LmerResult | GlmerResult
) -> None:
    for name in ("converged", "pirls_converged"):
        flag = getattr(fitted, name, True)
        if not isinstance(flag, bool | np.bool_) or not flag:
            raise ValueError(f"Power simulation refit did not converge: {name}")
    validate_finite_real("beta", fitted.beta, original.beta.shape)
    validate_finite_real("theta", fitted.theta, original.theta.shape)
    validate_finite_real("random effects", fitted.u, (original.matrices.n_random,))
    validate_finite_real("deviance", fitted.deviance, ())
    validate_finite_real("sigma", fitted.sigma, ())
    if fitted.sigma <= 0:
        raise ValueError("Power simulation refit sigma must be positive")


def extend(
    model: LmerResult | GlmerResult,
    along: str,
    n: int,
    data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Extend a dataset for power analysis.

    Creates an extended dataset by replicating or modifying the
    original data along a specified dimension.

    Parameters
    ----------
    model : LmerResult or GlmerResult
        A fitted mixed model.
    along : str
        What to extend:
        - A grouping factor name: Add more groups
        - "within": Add more observations per group
    n : int
        Target number (groups or observations per group).
    data : DataFrame, optional
        Original data. If None, uses model.model_frame().

    Returns
    -------
    DataFrame
        Extended dataset.

    Examples
    --------
    >>> extended_data = extend(model, along="Subject", n=30)
    >>> new_model = lmer("y ~ x + (1|Subject)", extended_data)
    """
    if isinstance(n, (int, np.integer)) and not isinstance(n, (bool, np.bool_)) and n < 1:
        raise ValueError("n must be at least 1")
    validate_simulation_count(n, "n")
    frame = _power_frame(
        model.model_frame() if data is None else data,
        response=model.formula.response,
        response_levels=model.matrices.response_levels,
    )
    groups = list(model.ngrps())
    if along == "within":
        if not groups:
            raise ValueError("Cannot extend within groups: model has no grouping factors")
        group = groups[0]
    elif along in groups:
        group = along
    else:
        raise ValueError(f"Unknown 'along' value: {along}. Use a grouping factor or 'within'.")
    extended, _ = _resample_frame(frame, group, n, within=along == "within", reduce=False)
    return extended


def _power_frame(
    data: Any, *, response: str | None = None, response_levels: tuple[Any, Any] | None = None
) -> pd.DataFrame:
    """Convert a stored model frame without requiring a Polars Arrow dependency."""
    import pandas as pd

    from mixedlm.utils.dataframe import (
        ensure_dataframe,
        get_categories,
        get_column_numpy,
        get_columns,
    )

    frame = ensure_dataframe(data)
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    converted = pd.DataFrame({name: get_column_numpy(frame, name) for name in get_columns(frame)})
    for name in converted.columns:
        dtype_name = str(frame[name].dtype)
        if "Categorical" in dtype_name or "Enum" in dtype_name:
            levels = (
                list(response_levels)
                if name == response and response_levels is not None
                else get_categories(frame, name)
            )
            converted[name] = pd.Categorical(converted[name], categories=levels)
    return converted


def _resample_frame(
    frame: pd.DataFrame, group: str, n: int, *, within: bool, reduce: bool
) -> tuple[pd.DataFrame, NDArray[np.intp]]:
    """Resize groups deterministically and retain source positions for row metadata."""
    import pandas as pd

    if group not in frame.columns:
        raise ValueError(f"Grouping factor {group!r} is not a model-frame column")
    grouped = list(frame.groupby(group, sort=False, observed=True, dropna=False).indices.values())
    if not grouped:
        raise ValueError("Power analysis requires a nonempty model frame")
    if within:
        rows = np.concatenate(
            [
                indices[np.arange(n if reduce else max(n, len(indices))) % len(indices)]
                for indices in grouped
            ]
        )
        return frame.iloc[rows].reset_index(drop=True), rows
    if n <= len(grouped):
        rows = np.sort(np.concatenate(grouped[:n])) if reduce else np.arange(len(frame))
        return frame.iloc[rows].reset_index(drop=True), rows

    categorical = isinstance(frame[group].dtype, pd.CategoricalDtype)
    labels = frame[group].cat.categories.tolist() if categorical else frame[group].unique().tolist()
    added_labels = _new_group_labels(labels, n - len(grouped))
    if categorical:
        frame[group] = frame[group].cat.add_categories(added_labels)
    pieces = [frame]
    positions = [np.arange(len(frame), dtype=np.intp)]
    for index, label in enumerate(added_labels):
        rows = grouped[index % len(grouped)]
        piece = frame.iloc[rows].copy()
        piece[group] = label
        if pd.api.types.is_unsigned_integer_dtype(frame[group].dtype):
            dtype = getattr(frame[group].dtype, "numpy_dtype", frame[group].dtype)
            if label <= np.iinfo(dtype).max:
                # A signed/unsigned concatenation can choose float64 and merge
                # distinct uint64 identifiers above its exact integer range.
                piece[group] = piece[group].astype(frame[group].dtype)
        pieces.append(piece)
        positions.append(rows)
    resized = pd.concat(pieces, ignore_index=True)
    if categorical:
        resized[group] = pd.Categorical(resized[group], dtype=frame[group].dtype)
    return resized, np.concatenate(positions)


def _resize_power_model(
    model: LmerResult | GlmerResult, group: str, n: int, *, within: bool = False
) -> LmerResult | GlmerResult:
    from mixedlm.matrices.design import build_random_matrix

    if model.matrices.frame is None:
        raise ValueError("Sample-size power curves require a stored model frame")
    frame, rows = _resample_frame(
        _power_frame(
            model.matrices.frame,
            response=model.formula.response,
            response_levels=model.matrices.response_levels,
        ),
        group,
        n,
        within=within,
        reduce=True,
    )
    source = model.matrices
    Z, structures = build_random_matrix(
        model.formula, frame, contrasts=source.contrasts, category_levels=source.category_levels
    )
    # Reuse fitted encodings and rank reduction, including fixed factors absent
    # from a smaller design. Only a grouping predictor needs to be re-encoded.
    X = (
        model._prediction_fixed_matrix(frame)
        if group in model.formula.fixed_variables
        else source.X[rows]
    )
    matrices = replace(
        source,
        y=source.y[rows],
        X=X,
        Z=Z,
        random_structures=structures,
        n_obs=len(rows),
        n_random=Z.shape[1],
        weights=source.weights[rows],
        offset=source.offset[rows],
        trials=None if source.trials is None else source.trials[rows],
        frame=frame,
        na_info=None,
    )
    # Preserve pilot parameters rather than re-estimating the alternative from
    # repeated pilot responses. Fresh result objects also discard fitted caches.
    return replace(
        model,
        matrices=matrices,
        beta=model.beta.copy(),
        theta=model.theta.copy(),
        u=np.zeros(Z.shape[1]),
    )


def powerCurve(
    model: LmerResult | GlmerResult,
    test: Callable[[LmerResult | GlmerResult], bool] | str | None = None,
    along: str = "n_groups",
    values: list[int | float] | None = None,
    nsim: int = _DEFAULT_NSIM_CURVE,
    alpha: float = _DEFAULT_ALPHA,
    seed: int | None = None,
    verbose: bool = False,
) -> PowerCurveResult:
    """Compute power across a range of values.

    Parameters
    ----------
    model : LmerResult or GlmerResult
        A fitted mixed model.
    test : callable or str, optional
        Test function or parameter name. If omitted for a named coefficient
        curve, tests that coefficient; otherwise tests the first non-intercept.
    along : str, default "n_groups"
        What to vary:
        - "n_groups": Number of groups in the first grouping factor
        - A grouping factor name: Number of groups in that factor
        - "within": Observations per group in the first grouping factor
        - "effect_size": Effect size (multiplier)
        - A fixed coefficient name: Set that coefficient to each absolute value
    values : list, optional
        Values to test. Sample sizes must be positive integers. Smaller designs
        select the first observed groups or rows; larger designs cycle pilot
        group or row templates. Pilot coefficients, covariance, prior weights,
        offsets, contrasts, and binomial trial counts are retained. If None,
        uses values around the current size or coefficient.
    nsim : int, default 500
        Simulations per point.
    alpha : float, default 0.05
        Significance level.
    seed : int, optional
        Random seed.
    verbose : bool, default False
        Print progress.

    Returns
    -------
    PowerCurveResult
        Object containing power curve data, plotting methods, and per-point
        PowerResult objects in ``results`` with fit-failure counts.

    Examples
    --------
    >>> curve = powerCurve(model, test="x", along="n_groups", values=[10, 20, 30, 40])
    >>> print(curve)
    >>> curve.plot()
    """
    _validate_power_options(nsim, alpha)
    groups = list(model.ngrps())
    within = along == "within"
    group = groups[0] if along in ("n_groups", "within") and groups else along
    design_curve = along in ("n_groups", "within") or along in groups
    if design_curve:
        if not groups:
            raise ValueError("Sample-size power curves require a grouping factor")
        if model.matrices.frame is None:
            raise ValueError("Sample-size power curves require a stored model frame")
        if group not in model.matrices.frame.columns:
            raise ValueError(f"Grouping factor {group!r} is not a model-frame column")
    elif along != "effect_size" and along not in model.matrices.fixed_names:
        raise ValueError(f"Unknown 'along' value: {along}")

    parameter = None
    if not design_curve:
        parameter = (
            (test if isinstance(test, str) else _default_parameter(model))
            if along == "effect_size"
            else along
        )
        _check_unique_coefficient_names(
            model.matrices.fixed_names,
            [parameter],
            alternative="Select an unambiguous coefficient name.",
        )
        if parameter not in model.matrices.fixed_names:
            raise ValueError(
                f"Parameter '{parameter}' not found. Available: {model.matrices.fixed_names}"
            )

    if values is None:
        if design_curve:
            if within:
                frame = _power_frame(model.matrices.frame)
                current = int(frame.groupby(group, observed=True).size().min())
                minimum = 1
            else:
                current = model.ngrps()[group]
                minimum = _MIN_GROUPS
            values = [max(minimum, current // 2), current, max(1, int(current * 1.5)), current * 2]
        elif along == "effect_size":
            values = list(_DEFAULT_EFFECT_SIZE_VALUES)
        else:
            assert parameter is not None
            index = model.matrices.fixed_names.index(parameter)
            values = [float(model.beta[index]) * scale for scale in _DEFAULT_EFFECT_SIZE_VALUES]
    values = list(values)
    if not values:
        raise ValueError("values must contain at least one curve point")
    for value in values:
        if design_curve:
            validate_simulation_count(cast(int, value), "sample size")
        elif isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError("effect values must be finite real numbers")

    results = []
    for i, value in enumerate(values):
        if verbose:
            print(f"Computing power for {along}={value} ({i + 1}/{len(values)})")
        iter_seed = seed + i * _SEED_OFFSET_MULTIPLIER if seed is not None else None
        if design_curve:
            modified_model = _resize_power_model(model, group, int(value), within=within)
        else:
            modified_model = _scale_effect(
                model, float(value), parameter, absolute=along != "effect_size"
            )
        curve_test = parameter if test is None and not design_curve else test
        result = powerSim(modified_model, test=curve_test, nsim=nsim, alpha=alpha, seed=iter_seed)
        if design_curve:
            result.n_groups = modified_model.ngrps()[group]
        results.append(result)

    return PowerCurveResult(
        values=values,
        powers=[result.power for result in results],
        ci_lowers=[result.ci_lower for result in results],
        ci_uppers=[result.ci_upper for result in results],
        along=along,
        n_simulations=nsim,
        results=tuple(results),
    )


def _scale_effect(
    model: LmerResult | GlmerResult,
    scale: float,
    param: str | None = None,
    *,
    absolute: bool = False,
) -> LmerResult | GlmerResult:
    """Change a generating coefficient without copying the design or fitted caches."""
    param = _default_parameter(model) if param is None else param
    _check_unique_coefficient_names(
        model.matrices.fixed_names, [param], alternative="Select an unambiguous coefficient name."
    )
    idx = model.matrices.fixed_names.index(param)
    beta = model.beta.copy()
    beta[idx] = scale if absolute else model.beta[idx] * scale
    return replace(model, beta=beta)


__all__ = [
    "PowerResult",
    "PowerCurveResult",
    "powerSim",
    "powerCurve",
    "extend",
]
