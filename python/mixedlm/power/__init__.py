from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import stats

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
_DEFAULT_N_GROUPS_VALUES = [10, 20, 30, 40]
_SEED_OFFSET_MULTIPLIER = 1000
_FIGURE_SIZE = (8, 5)


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
        return "\n".join(lines)


@dataclass
class PowerCurveResult:
    values: list[int | float]
    powers: list[float]
    ci_lowers: list[float]
    ci_uppers: list[float]
    along: str
    n_simulations: int

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

    param_names = result.matrices.fixed_names
    if param not in param_names:
        raise ValueError(f"Parameter '{param}' not found. Available: {param_names}")

    idx = param_names.index(param)
    se = np.sqrt(vcov[idx, idx])
    z_val = beta[idx] / se if se > 0 else 0.0

    p_val = 2 * stats.norm.sf(np.abs(z_val))
    return bool(p_val < alpha)


def _binomial_score_interval(
    n_successes: int,
    n_trials: int,
    alpha: float,
) -> tuple[float, float]:
    """Compute a Wilson score interval for a binomial proportion."""
    proportion = n_successes / n_trials
    z_critical = float(stats.norm.ppf(1.0 - alpha / 2.0))
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
        If None, tests the second fixed effect (first non-intercept).
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

    Examples
    --------
    >>> result = lmer("y ~ x + (1|group)", data)
    >>> power = powerSim(result, test="x", nsim=500)
    >>> print(power)
    """
    if isinstance(nsim, bool) or not isinstance(nsim, (int, np.integer)) or nsim < 1:
        raise ValueError("nsim must be a positive integer")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")

    param_names = model.matrices.fixed_names
    if not param_names:
        raise ValueError("model has no fixed effects to test")

    rng = np.random.default_rng(seed)

    def make_test_func(param: str, a: float) -> Callable[[LmerResult | GlmerResult], bool]:
        def test_fn(m: LmerResult | GlmerResult) -> bool:
            return _default_test(m, param, a)

        return test_fn

    test_param: str | None = None
    if test is None:
        test_param = param_names[1] if len(param_names) > 1 else param_names[0]
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
            if test_func(fit_sim):
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
    )


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
    import pandas as pd

    if data is None:
        data = model.model_frame()

    data = data.copy()

    grp_factors = list(model.ngrps().keys())

    if along in grp_factors:
        current_groups = data[along].unique()
        n_current = len(current_groups)

        if n <= n_current:
            return data

        n_to_add = n - n_current
        template = data.groupby(along).first().reset_index()

        new_data_list = [data]
        for i in range(n_to_add):
            new_group = template.copy()
            new_group[along] = f"new_group_{i + 1}"
            new_data_list.append(new_group)

        return pd.concat(new_data_list, ignore_index=True)

    elif along == "within":
        obs_per_group = data.groupby(grp_factors[0]).size()
        current_n = int(obs_per_group.mean())

        if n <= current_n:
            return data

        factor = n // current_n
        if factor <= 1:
            return data

        new_data_list = [data]
        for _ in range(factor - 1):
            new_data_list.append(data.copy())

        return pd.concat(new_data_list, ignore_index=True)

    else:
        raise ValueError(f"Unknown 'along' value: {along}. Use a grouping factor or 'within'.")


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
        Test function or parameter name.
    along : str, default "n_groups"
        What to vary:
        - "n_groups": Number of groups
        - "effect_size": Effect size (multiplier)
        - A parameter name: Vary that parameter
    values : list, optional
        Values to test. If None, uses sensible defaults.
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
        Object containing power curve data and plotting methods.

    Examples
    --------
    >>> curve = powerCurve(model, test="x", along="n_groups", values=[10, 20, 30, 40])
    >>> print(curve)
    >>> curve.plot()
    """
    base_seed = seed if seed is not None else None

    if values is None:
        if along == "n_groups":
            grps = model.ngrps()
            if grps:
                current = list(grps.values())[0]
                values = [
                    max(_MIN_GROUPS, current // 2),
                    current,
                    int(current * 1.5),
                    current * 2,
                ]
            else:
                values = list(_DEFAULT_N_GROUPS_VALUES)
        elif along == "effect_size":
            values = list(_DEFAULT_EFFECT_SIZE_VALUES)
        else:
            values = list(_DEFAULT_EFFECT_SIZE_VALUES)

    powers = []
    ci_lowers = []
    ci_uppers = []

    for i, v in enumerate(values):
        if verbose:
            print(f"Computing power for {along}={v} ({i + 1}/{len(values)})")

        iter_seed = base_seed + i * _SEED_OFFSET_MULTIPLIER if base_seed is not None else None

        if along == "effect_size":
            test_param = test if isinstance(test, str) else None
            modified_model = _scale_effect(model, v, test_param)
            result = powerSim(
                modified_model,
                test=test,
                nsim=nsim,
                alpha=alpha,
                seed=iter_seed,
                verbose=False,
            )
        else:
            result = powerSim(
                model,
                test=test,
                nsim=nsim,
                alpha=alpha,
                seed=iter_seed,
                verbose=False,
            )

        powers.append(result.power)
        ci_lowers.append(result.ci_lower)
        ci_uppers.append(result.ci_upper)

    return PowerCurveResult(
        values=values,
        powers=powers,
        ci_lowers=ci_lowers,
        ci_uppers=ci_uppers,
        along=along,
        n_simulations=nsim,
    )


def _scale_effect(
    model: LmerResult | GlmerResult,
    scale: float,
    param: str | None = None,
) -> LmerResult | GlmerResult:
    """Scale the effect size of a parameter."""
    from copy import deepcopy

    model_copy = deepcopy(model)

    if param is None:
        param_names = model.matrices.fixed_names
        param = param_names[1] if len(param_names) > 1 else param_names[0]

    idx = model.matrices.fixed_names.index(param)
    model_copy.beta[idx] = model.beta[idx] * scale

    return model_copy


__all__ = [
    "PowerResult",
    "PowerCurveResult",
    "powerSim",
    "powerCurve",
    "extend",
]
