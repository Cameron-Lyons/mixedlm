from __future__ import annotations

import os
import sys
from collections.abc import Callable, Generator, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import closing
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import partial
from numbers import Integral
from threading import local
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from mixedlm.utils.names import _check_unique_coefficient_names
from mixedlm.utils.random import RandomSeed, random_seeds, random_stream, validate_simulation_count
from mixedlm.utils.simulation import simulate_random_effects

if TYPE_CHECKING:
    from mixedlm.estimation.laplace import GLMMOptimizationResult
    from mixedlm.estimation.reml import LMMOptimizer, OptimizationResult
    from mixedlm.families.base import Family
    from mixedlm.matrices.design import ModelMatrices
    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult
    from mixedlm.models.nlmer import NlmerResult


_BOOTSTRAP_CI_METHODS = ("percentile", "basic", "normal")


def _validate_ci_options(level: float, method: str) -> None:
    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("level must be strictly between 0 and 1")
    if method not in _BOOTSTRAP_CI_METHODS:
        raise ValueError(f"Unknown method: {method}")


_FailureStage = Literal["simulation", "refit", "convergence", "validation"]
_FAILURE_STAGES: tuple[_FailureStage, ...] = ("simulation", "refit", "convergence", "validation")


@dataclass(frozen=True)
class BootstrapFailure:
    """A failed bootstrap replicate, indexed by its zero-based sample row.

    ``stage`` is simulation, refit, convergence, or validation. Exceptions are
    stored as type names and messages, without retaining exception objects or
    tracebacks. Convergence and validation checks report their own ValueError
    (or AttributeError if a required refit attribute is missing).
    """

    index: int
    stage: _FailureStage
    exception_type: str
    message: str


@dataclass
class _BootstrapOutcome:
    index: int
    fixed: NDArray[np.float64] | None = None
    theta: NDArray[np.float64] | None = None
    sigma: float | None = None
    failure: BootstrapFailure | None = None


def _bootstrap_failure(index: int, stage: _FailureStage, error: Exception) -> BootstrapFailure:
    try:
        message = str(error)
    except Exception:
        message = "Exception message could not be formatted"
    return BootstrapFailure(index, stage, type(error).__name__, message)


def _require_bootstrap_convergence(fitted: Any, inner: str | None) -> None:
    names = ("converged",) if inner is None else ("converged", inner)
    failed = []
    for name in names:
        flag = getattr(fitted, name)
        if not isinstance(flag, bool | np.bool_) or not flag:
            failed.append(name)
    if failed:
        message = "Bootstrap refit did not converge: " + ", ".join(failed)
        detail = getattr(fitted, "message", "")
        if isinstance(detail, str) and detail:
            message += f" ({detail})"
        raise ValueError(message)


def _bootstrap_sample_vector(values: Any, size: int, name: str) -> NDArray[np.float64]:
    array = np.asarray(values)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if array.dtype.kind not in "biuf" or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite real numbers")
    with np.errstate(over="raise", invalid="raise"):
        return array.astype(np.float64, copy=True)


def _bootstrap_sample_scale(value: Any) -> float:
    array = np.asarray(value)
    if (
        array.shape != ()
        or array.dtype.kind not in "biuf"
        or not np.isfinite(array)
        or array <= 0.0
    ):
        raise ValueError("sigma must be a finite positive scalar")
    scale = float(array)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("sigma must fit in a finite positive float64")
    return scale


def _bootstrap_simulation(
    index: int, simulate: Callable[[], Any], n_obs: int
) -> NDArray[np.float64] | BootstrapFailure:
    try:
        # Own the response storage before asynchronous process serialization.
        return _bootstrap_sample_vector(simulate(), n_obs, "Simulated response")
    except Exception as error:
        return _bootstrap_failure(index, "simulation", error)


def _bootstrap_refit(
    index: int,
    response: NDArray[np.floating] | BootstrapFailure,
    refit: Callable[[NDArray[np.floating]], Any],
    fixed_name: str,
    n_fixed: int,
    n_theta: int,
    *,
    inner: str | None = None,
    has_scale: bool = True,
) -> _BootstrapOutcome:
    if isinstance(response, BootstrapFailure):
        return _BootstrapOutcome(index, failure=response)
    stage: _FailureStage = "refit"
    try:
        fitted = refit(response)
        stage = "convergence"
        _require_bootstrap_convergence(fitted, inner)
        stage = "validation"
        # Validate every component before publishing any part of the sample.
        fixed = _bootstrap_sample_vector(getattr(fitted, fixed_name), n_fixed, fixed_name)
        theta = _bootstrap_sample_vector(fitted.theta, n_theta, "theta")
        sigma = _bootstrap_sample_scale(fitted.sigma) if has_scale else None
        return _BootstrapOutcome(index, fixed, theta, sigma)
    except Exception as error:
        return _BootstrapOutcome(index, failure=_bootstrap_failure(index, stage, error))


def _store_bootstrap_sample(
    sample: _BootstrapOutcome,
    fixed: NDArray[np.floating],
    theta: NDArray[np.floating],
    sigma: NDArray[np.floating] | None,
    failures: list[BootstrapFailure],
) -> None:
    if sample.failure is not None:
        failures.append(sample.failure)
        return
    assert sample.fixed is not None and sample.theta is not None
    fixed[sample.index, :] = sample.fixed
    theta[sample.index, :] = sample.theta
    if sigma is not None:
        assert sample.sigma is not None
        sigma[sample.index] = sample.sigma


def _finite_bootstrap_samples(
    samples: NDArray[np.floating],
    column: int,
) -> NDArray[np.floating]:
    values = samples[:, column]
    return values[np.isfinite(values)]


def _bootstrap_ci(
    samples: NDArray[np.floating],
    original: NDArray[np.floating],
    names: list[str],
    level: float,
    method: str,
) -> dict[str, tuple[float, float]]:
    _validate_ci_options(level, method)
    _check_unique_coefficient_names(
        names, alternative="Use bootCI() for one row per parameter in sample-column order."
    )
    alpha = 1.0 - float(level)
    lower_percentile = 100.0 * alpha / 2.0
    upper_percentile = 100.0 * (1.0 - alpha / 2.0)
    z_critical = stats.norm.isf(alpha / 2.0) if method == "normal" else None
    result: dict[str, tuple[float, float]] = {}

    for i, name in enumerate(names):
        parameter_samples = _finite_bootstrap_samples(samples, i)
        if len(parameter_samples) < 2:
            result[name] = (np.nan, np.nan)
            continue

        if method == "percentile":
            lower, upper = np.percentile(
                parameter_samples,
                [lower_percentile, upper_percentile],
            )
        elif method == "basic":
            upper_sample, lower_sample = np.percentile(
                parameter_samples,
                [upper_percentile, lower_percentile],
            )
            lower = 2.0 * original[i] - upper_sample
            upper = 2.0 * original[i] - lower_sample
        else:
            standard_error = np.std(parameter_samples, ddof=1)
            bias = np.mean(parameter_samples) - original[i]
            center = original[i] - bias
            assert z_critical is not None
            lower = center - z_critical * standard_error
            upper = center + z_critical * standard_error

        result[name] = (float(lower), float(upper))

    return result


def _bootstrap_se(
    samples: NDArray[np.floating],
    names: list[str],
) -> dict[str, float]:
    _check_unique_coefficient_names(
        names, alternative="Use bootCI() for standard errors in sample-column order."
    )
    result: dict[str, float] = {}
    for i, name in enumerate(names):
        parameter_samples = _finite_bootstrap_samples(samples, i)
        result[name] = (
            float(np.std(parameter_samples, ddof=1)) if len(parameter_samples) > 1 else np.nan
        )
    return result


def _bootstrap_summary(
    n_boot: int,
    n_failed: int,
    samples: NDArray[np.floating],
    original: NDArray[np.floating],
    names: list[str],
    failures: tuple[BootstrapFailure, ...],
) -> str:
    lines = [
        f"Parametric bootstrap with {n_boot} samples ({n_failed} failed)",
        "",
        "Fixed effects bootstrap statistics:",
        "             Original    Mean       Bias     Std.Err",
    ]

    for i, name in enumerate(names):
        parameter_samples = _finite_bootstrap_samples(samples, i)
        if len(parameter_samples) == 0:
            continue
        mean = np.mean(parameter_samples)
        bias = mean - original[i]
        standard_error = np.std(parameter_samples, ddof=1) if len(parameter_samples) > 1 else np.nan
        lines.append(
            f"{name:12} {original[i]:10.4f} {mean:10.4f} {bias:10.4f} {standard_error:10.4f}"
        )

    if failures:
        counts = {stage: 0 for stage in _FAILURE_STAGES}
        for failure in failures:
            counts[failure.stage] += 1
        lines.extend(["", "Failed samples by stage:"])
        lines.extend(f"  {stage}: {count}" for stage, count in counts.items() if count)

    return "\n".join(lines)


@dataclass
class BootstrapResult:
    n_boot: int
    beta_samples: NDArray[np.floating]
    theta_samples: NDArray[np.floating]
    sigma_samples: NDArray[np.floating] | None
    fixed_names: list[str]
    original_beta: NDArray[np.floating]
    original_theta: NDArray[np.floating]
    original_sigma: float | None
    n_failed: int
    failures: tuple[BootstrapFailure, ...] = ()

    def ci(
        self,
        level: float = 0.95,
        method: str = "percentile",
    ) -> dict[str, tuple[float, float]]:
        return _bootstrap_ci(
            self.beta_samples,
            self.original_beta,
            self.fixed_names,
            level,
            method,
        )

    def se(self) -> dict[str, float]:
        return _bootstrap_se(self.beta_samples, self.fixed_names)

    def summary(self) -> str:
        return _bootstrap_summary(
            self.n_boot,
            self.n_failed,
            self.beta_samples,
            self.original_beta,
            self.fixed_names,
            self.failures,
        )


_BootstrapSample = TypeVar("_BootstrapSample")
_bootstrap_worker_state = local()


def _bootstrap_worker_count(n_jobs: int, n_boot: int) -> int:
    if isinstance(n_jobs, bool | np.bool_) or not isinstance(n_jobs, Integral):
        raise TypeError("n_jobs must be -1 or a positive integer")
    if n_jobs != -1 and n_jobs < 1:
        raise ValueError("n_jobs must be -1 or a positive integer")
    if n_jobs == -1:
        workers = os.cpu_count() or 1
        if sys.platform == "win32":
            workers = min(workers, 61)
    else:
        workers = int(n_jobs)
    return min(workers, n_boot)


def _initialize_bootstrap_worker(
    worker: Callable[[tuple[Any, ...]], Any],
    data: tuple[Any, ...],
) -> None:
    # Thread-local storage also isolates independent pools in threaded callers.
    if worker is _lmer_bootstrap_worker:
        from mixedlm.estimation.reml import LMMOptimizer

        # Construct native state after process startup; it is never pickled.
        data = (*data, LMMOptimizer(data[0], REML=data[-1], use_rust=True))
    _bootstrap_worker_state.worker = worker
    _bootstrap_worker_state.data = data


def _run_bootstrap_task(task: tuple[int, Any]) -> Any:
    return _bootstrap_worker_state.worker((*task, *_bootstrap_worker_state.data))


def _parallel_bootstrap_samples(
    worker: Callable[[tuple[Any, ...]], _BootstrapSample],
    data: tuple[Any, ...],
    seeds: NDArray[np.integer],
    workers: int,
) -> Generator[_BootstrapSample, None, None]:
    tasks = ((index, int(seed)) for index, seed in enumerate(seeds))
    yield from _parallel_bootstrap_tasks(worker, data, tasks, workers)


def _parallel_bootstrap_tasks(
    worker: Callable[[tuple[Any, ...]], _BootstrapSample],
    data: tuple[Any, ...],
    tasks: Iterable[tuple[int, Any]],
    workers: int,
) -> Generator[_BootstrapSample, None, None]:
    """Consume a bounded stream of seed or response tasks using shared worker data."""
    tasks = iter(tasks)
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_bootstrap_worker,
        initargs=(worker, data),
    ) as executor:
        pending: set[Future[_BootstrapSample]] = set()
        try:
            for _ in range(2 * workers):
                task = next(tasks, None)
                if task is None:
                    break
                pending.add(executor.submit(_run_bootstrap_task, task))
            while pending:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                while completed:
                    future = completed.pop()
                    yield future.result()
                    following = next(tasks, None)
                    if following is not None:
                        pending.add(executor.submit(_run_bootstrap_task, following))
        except BaseException:
            # Interruptions and pool failures must not run the remaining bootstrap.
            for future in pending:
                future.cancel()
            raise


def _refit_lmer_response(
    matrices: ModelMatrices,
    response: NDArray[np.floating],
    theta: NDArray[np.floating],
    REML: bool,
    *,
    optimizer: LMMOptimizer | None = None,
) -> OptimizationResult:
    """Refit an LMM response against the original validated design matrices."""
    from mixedlm.estimation.reml import LMMOptimizer

    if optimizer is None:
        bootstrap_matrices = replace(matrices, y=np.ascontiguousarray(response))
        optimizer = LMMOptimizer(bootstrap_matrices, REML=REML, use_rust=True)
    else:
        optimizer = optimizer.with_response(response)
    return optimizer.optimize(start=theta)


def _refit_glmer_response(
    matrices: ModelMatrices,
    response: NDArray[np.floating],
    theta: NDArray[np.floating],
    family: Family,
    nAGQ: int,
    pirls_maxiter: int | None = None,
    pirls_tol: float = 1e-6,
) -> GLMMOptimizationResult:
    """Refit a GLMM response against the original validated design matrices."""
    from mixedlm.estimation.laplace import GLMMOptimizer

    bootstrap_matrices = replace(matrices, y=np.ascontiguousarray(response))
    optimizer = GLMMOptimizer(
        bootstrap_matrices,
        family,
        nAGQ=nAGQ,
        pirls_maxiter=pirls_maxiter,
        pirls_tol=pirls_tol,
    )
    return optimizer.optimize(start=theta)


def _lmer_bootstrap_worker(
    args: tuple[Any, ...],
) -> _BootstrapOutcome:
    (
        boot_idx,
        seed,
        matrices,
        beta,
        theta,
        sigma,
        REML,
        *prepared,
    ) = args

    rng = np.random.RandomState(seed)

    response = _bootstrap_simulation(
        boot_idx,
        lambda: _simulate_lmer_components(matrices, beta, theta, sigma, rng),
        matrices.n_obs,
    )
    return _bootstrap_refit(
        boot_idx,
        response,
        lambda y: _refit_lmer_response(
            matrices, y, theta, REML, optimizer=prepared[0] if prepared else None
        ),
        "beta",
        matrices.n_fixed,
        len(theta),
    )


def _glmer_bootstrap_worker(args: tuple[Any, ...]) -> _BootstrapOutcome:
    (
        boot_idx,
        seed,
        matrices,
        beta,
        theta,
        family,
        nAGQ,
        pirls_maxiter,
        pirls_tol,
    ) = args

    rng = np.random.RandomState(seed)

    try:
        # Each task previously deserialized its own family; preserve that isolation.
        family = deepcopy(family)
    except Exception as error:
        return _BootstrapOutcome(
            boot_idx, failure=_bootstrap_failure(boot_idx, "simulation", error)
        )
    response = _bootstrap_simulation(
        boot_idx,
        lambda: _simulate_glmer_components(matrices, beta, theta, family, rng),
        matrices.n_obs,
    )
    return _bootstrap_refit(
        boot_idx,
        response,
        lambda y: _refit_glmer_response(matrices, y, theta, family, nAGQ, pirls_maxiter, pirls_tol),
        "beta",
        matrices.n_fixed,
        len(theta),
        inner="pirls_converged",
        has_scale=False,
    )


def _prepare_lmer_worker_data(result: LmerResult) -> dict[str, Any]:
    return {
        "matrices": replace(result.matrices, frame=None, na_info=None),
        "beta": result.beta.copy(),
        "theta": result.theta.copy(),
        "sigma": result.sigma,
        "REML": result.REML,
    }


def _prepare_glmer_worker_data(result: GlmerResult) -> dict[str, Any]:
    return {
        "matrices": replace(result.matrices, frame=None, na_info=None),
        "beta": result.beta.copy(),
        "theta": result.theta.copy(),
        "family": result.family,
        "nAGQ": result.nAGQ,
        "pirls_maxiter": result.pirls_maxiter,
        "pirls_tol": result.pirls_tol,
    }


def bootstrap_lmer(
    result: LmerResult,
    n_boot: int = 1000,
    seed: RandomSeed = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> BootstrapResult:
    """Simulate and refit an LMM, excluding unsuccessful or invalid refits.

    Failed replicates remain entirely NaN and increment ``n_failed`` in both
    serial and parallel execution. Details are recorded in ``failures``.
    Confidence bounds and standard errors require at least two valid samples
    per parameter.

    ``n_jobs`` must be a positive integer or -1 for available CPUs. Parallel
    workers reuse the fitted design and keep only a bounded number of tasks
    outstanding; the worker count never exceeds ``n_boot``. Weighted design
    products are prepared once per serial call or parallel worker.
    """
    validate_simulation_count(n_boot, "n_boot")
    workers = _bootstrap_worker_count(n_jobs, n_boot)
    p = result.matrices.n_fixed
    n_theta = len(result.theta)

    beta_samples = np.full((n_boot, p), np.nan)
    theta_samples = np.full((n_boot, n_theta), np.nan)
    sigma_samples = np.full(n_boot, np.nan)

    rng = random_stream(seed, legacy=False)
    seeds = random_seeds(rng, n_boot)

    failures: list[BootstrapFailure] = []

    if n_jobs == 1:
        from mixedlm.estimation.reml import LMMOptimizer

        optimizer = LMMOptimizer(result.matrices, REML=result.REML, use_rust=True)

        for b in range(n_boot):
            if verbose and (b + 1) % 100 == 0:
                print(f"Bootstrap iteration {b + 1}/{n_boot}")

            simulation_rng = np.random.RandomState(int(seeds[b]))

            response = _bootstrap_simulation(
                b, partial(_simulate_lmer, result, simulation_rng), result.matrices.n_obs
            )
            sample = _bootstrap_refit(
                b,
                response,
                lambda y: _refit_lmer_response(
                    result.matrices, y, result.theta, result.REML, optimizer=optimizer
                ),
                "beta",
                p,
                n_theta,
            )
            _store_bootstrap_sample(sample, beta_samples, theta_samples, sigma_samples, failures)
    else:
        worker_data = _prepare_lmer_worker_data(result)
        with closing(
            _parallel_bootstrap_samples(
                _lmer_bootstrap_worker, tuple(worker_data.values()), seeds, workers
            )
        ) as samples:
            for completed, sample in enumerate(samples, start=1):
                if verbose and completed % 100 == 0:
                    print(f"Bootstrap iteration {completed}/{n_boot}")
                _store_bootstrap_sample(
                    sample, beta_samples, theta_samples, sigma_samples, failures
                )

    return BootstrapResult(
        n_boot=n_boot,
        beta_samples=beta_samples,
        theta_samples=theta_samples,
        sigma_samples=sigma_samples,
        fixed_names=result.matrices.fixed_names,
        original_beta=result.beta,
        original_theta=result.theta,
        original_sigma=result.sigma,
        n_failed=len(failures),
        failures=tuple(sorted(failures, key=lambda failure: failure.index)),
    )


def _simulate_lmer(result: LmerResult, rng: Any | None = None) -> NDArray[np.floating]:
    return _simulate_lmer_components(result.matrices, result.beta, result.theta, result.sigma, rng)


def _simulate_lmer_components(
    matrices: ModelMatrices,
    beta: NDArray[np.floating],
    theta: NDArray[np.floating],
    sigma: float,
    rng: Any | None = None,
) -> NDArray[np.floating]:
    rng = np.random if rng is None else rng
    n = matrices.n_obs
    q = matrices.n_random

    fixed_part = matrices.X @ beta + matrices.offset

    if q > 0:
        u_new = simulate_random_effects(theta, matrices.random_structures, sigma, rng=rng)
        random_part = matrices.Z @ u_new
    else:
        random_part = np.zeros(n)

    noise = rng.standard_normal(n) * sigma / np.sqrt(matrices.weights)

    return fixed_part + random_part + noise


def bootstrap_glmer(
    result: GlmerResult,
    n_boot: int = 1000,
    seed: RandomSeed = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> BootstrapResult:
    """Simulate and refit a GLMM, requiring outer and inner convergence.

    Failed or invalid replicates remain entirely NaN and increment ``n_failed``
    in both serial and parallel execution. Details are recorded in ``failures``.
    Confidence bounds and standard errors require at least two valid samples
    per parameter.

    ``n_jobs`` must be a positive integer or -1 for available CPUs. Parallel
    workers reuse the fitted design and keep only a bounded number of tasks
    outstanding; the worker count never exceeds ``n_boot``.
    """
    validate_simulation_count(n_boot, "n_boot")
    workers = _bootstrap_worker_count(n_jobs, n_boot)
    p = result.matrices.n_fixed
    n_theta = len(result.theta)

    beta_samples = np.full((n_boot, p), np.nan)
    theta_samples = np.full((n_boot, n_theta), np.nan)

    rng = random_stream(seed, legacy=False)
    seeds = random_seeds(rng, n_boot)

    failures: list[BootstrapFailure] = []

    if n_jobs == 1:
        for b in range(n_boot):
            if verbose and (b + 1) % 100 == 0:
                print(f"Bootstrap iteration {b + 1}/{n_boot}")

            simulation_rng = np.random.RandomState(int(seeds[b]))

            response = _bootstrap_simulation(
                b, partial(_simulate_glmer, result, simulation_rng), result.matrices.n_obs
            )
            sample = _bootstrap_refit(
                b,
                response,
                lambda y: _refit_glmer_response(
                    result.matrices,
                    y,
                    result.theta,
                    result.family,
                    result.nAGQ,
                    result.pirls_maxiter,
                    result.pirls_tol,
                ),
                "beta",
                p,
                n_theta,
                inner="pirls_converged",
                has_scale=False,
            )
            _store_bootstrap_sample(sample, beta_samples, theta_samples, None, failures)
    else:
        worker_data = _prepare_glmer_worker_data(result)
        with closing(
            _parallel_bootstrap_samples(
                _glmer_bootstrap_worker, tuple(worker_data.values()), seeds, workers
            )
        ) as samples:
            for completed, sample in enumerate(samples, start=1):
                if verbose and completed % 100 == 0:
                    print(f"Bootstrap iteration {completed}/{n_boot}")
                _store_bootstrap_sample(sample, beta_samples, theta_samples, None, failures)

    return BootstrapResult(
        n_boot=n_boot,
        beta_samples=beta_samples,
        theta_samples=theta_samples,
        sigma_samples=None,
        fixed_names=result.matrices.fixed_names,
        original_beta=result.beta,
        original_theta=result.theta,
        original_sigma=None,
        n_failed=len(failures),
        failures=tuple(sorted(failures, key=lambda failure: failure.index)),
    )


def _simulate_glmer(result: GlmerResult, rng: Any | None = None) -> NDArray[np.floating]:
    return _simulate_glmer_components(
        result.matrices, result.beta, result.theta, result.family, rng
    )


def _simulate_glmer_components(
    matrices: ModelMatrices,
    beta: NDArray[np.floating],
    theta: NDArray[np.floating],
    family: Family,
    rng: Any | None = None,
) -> NDArray[np.floating]:
    rng = np.random if rng is None else rng
    q = matrices.n_random

    if q > 0:
        u_new = simulate_random_effects(theta, matrices.random_structures, rng=rng)
        eta = matrices.X @ beta + matrices.Z @ u_new + matrices.offset
    else:
        eta = matrices.X @ beta + matrices.offset

    mu = family.link.inverse(eta)
    if family.__class__.__name__ == "Binomial" and matrices.trials is not None:
        mu = family.clamp_mu(mu, eps=1e-6)
        trials = matrices.trials.astype(np.int64)
        successes = rng.binomial(trials, mu).astype(np.float64)
        return successes / trials
    return family.simulate(mu, rng=rng)


def bootMer(
    model: LmerResult | GlmerResult | NlmerResult,
    nsim: int = 1000,
    seed: RandomSeed = None,
    n_jobs: int = 1,
    verbose: bool = False,
    bootstrap_type: str = "parametric",
) -> BootstrapResult | NlmerBootstrapResult:
    """Model-based (semi-)parametric bootstrap for mixed models.

    This function provides an lme4-compatible interface for bootstrapping
    mixed models. It selects the linear, generalized, or nonlinear bootstrap
    function based on the model type.

    Parameters
    ----------
    model : LmerResult, GlmerResult, or NlmerResult
        A fitted mixed model.
    nsim : int, default 1000
        Number of bootstrap samples.
    seed : int, RandomState, or Generator, optional
        Local random seed or reusable stream for reproducibility.
    n_jobs : int, default 1
        Positive worker count or -1 for available CPUs, for all model types.
        The worker count never exceeds the number of replicates.
    verbose : bool, default False
        Print progress information.
    bootstrap_type : str, default "parametric"
        Type of bootstrap. Currently only "parametric" is supported.
        Parametric bootstrap simulates new responses from the fitted
        model and refits.

    Returns
    -------
    BootstrapResult or NlmerBootstrapResult
        Bootstrap results containing:
        - n_boot: Number of bootstrap samples
        - beta_samples: Fixed effects estimates from each sample
        - theta_samples: Variance parameter estimates from each sample
        - sigma_samples: Residual SD estimates (linear and nonlinear models)
        - failures: Ordered BootstrapFailure records for unsuccessful samples
        - Methods: ci(), se(), summary()

        Nonlinear models provide ``phi_samples`` in place of ``beta_samples``.

    Raises
    ------
    ValueError
        If an unsupported bootstrap type is requested.
    TypeError
        If model is not a supported type.

    Examples
    --------
    >>> result = lmer("Reaction ~ Days + (Days|Subject)", sleepstudy)
    >>> boot = bootMer(result, nsim=500, seed=42)
    >>> boot.ci(level=0.95)
    {'(Intercept)': (230.5, 270.3), 'Days': (7.5, 13.2)}
    >>> boot.se()
    {'(Intercept)': 9.8, 'Days': 1.4}

    >>> result = glmer("y ~ x + (1|group)", data, family=Binomial())
    >>> boot = bootMer(result, nsim=200)
    >>> print(boot.summary())

    Notes
    -----
    The parametric bootstrap:
    1. Simulates new response vectors from the fitted model
    2. Refits the model to each simulated dataset
    3. Collects the parameter estimates

    Refits must converge and return finite real estimates of the expected
    shapes, with a positive residual scale where applicable. Failed replicates
    remain entirely NaN and increment ``n_failed``. Confidence bounds and
    standard errors require at least two valid samples per parameter; inspect
    ``n_failed`` and ``failures`` before interpreting the results. Interval
    accuracy still depends on the fitted model and the number of successful replicates.

    See Also
    --------
    bootstrap_lmer : Bootstrap for linear mixed models.
    bootstrap_glmer : Bootstrap for generalized linear mixed models.
    bootstrap_nlmer : Bootstrap for nonlinear mixed models.
    confint : Confidence intervals (supports bootstrap method).
    """
    if bootstrap_type != "parametric":
        raise ValueError(
            f"Bootstrap type '{bootstrap_type}' not supported. Only 'parametric' is available."
        )

    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult
    from mixedlm.models.nlmer import NlmerResult

    if isinstance(model, LmerResult):
        return bootstrap_lmer(
            model,
            n_boot=nsim,
            seed=seed,
            n_jobs=n_jobs,
            verbose=verbose,
        )
    if isinstance(model, GlmerResult):
        return bootstrap_glmer(
            model,
            n_boot=nsim,
            seed=seed,
            n_jobs=n_jobs,
            verbose=verbose,
        )
    if isinstance(model, NlmerResult):
        return bootstrap_nlmer(
            model,
            n_boot=nsim,
            seed=seed,
            verbose=verbose,
            n_jobs=n_jobs,
        )
    raise TypeError(
        f"Model type {type(model).__name__} not supported. "
        "Use LmerResult, GlmerResult, or NlmerResult."
    )


@dataclass
class NlmerBootstrapResult:
    n_boot: int
    phi_samples: NDArray[np.floating]
    theta_samples: NDArray[np.floating]
    sigma_samples: NDArray[np.floating]
    param_names: list[str]
    original_phi: NDArray[np.floating]
    original_theta: NDArray[np.floating]
    original_sigma: float
    n_failed: int
    failures: tuple[BootstrapFailure, ...] = ()

    def ci(
        self,
        level: float = 0.95,
        method: str = "percentile",
    ) -> dict[str, tuple[float, float]]:
        return _bootstrap_ci(
            self.phi_samples,
            self.original_phi,
            self.param_names,
            level,
            method,
        )

    def se(self) -> dict[str, float]:
        return _bootstrap_se(self.phi_samples, self.param_names)

    def summary(self) -> str:
        return _bootstrap_summary(
            self.n_boot,
            self.n_failed,
            self.phi_samples,
            self.original_phi,
            self.param_names,
            self.failures,
        )


def _nlmer_bootstrap_refit(
    result: NlmerResult,
    response: NDArray[np.floating] | BootstrapFailure,
    *,
    index: int = 0,
) -> _BootstrapOutcome:
    return _bootstrap_refit(
        index,
        response,
        result.refit,
        "phi",
        len(result.phi),
        len(result.theta),
        inner="pnls_converged",
    )


def _nlmer_bootstrap_worker(args: tuple[Any, ...]) -> _BootstrapOutcome:
    index, response, result = args
    if isinstance(response, BootstrapFailure):
        return _BootstrapOutcome(index, failure=response)
    try:
        # A custom model's per-fit caches must not leak into later worker tasks.
        result = replace(result, model=deepcopy(result.model))
    except Exception as error:
        return _BootstrapOutcome(index, failure=_bootstrap_failure(index, "refit", error))
    return _nlmer_bootstrap_refit(result, response, index=index)


def _nlmer_bootstrap_responses(
    result: NlmerResult,
    n_boot: int,
    rng: Any,
) -> Generator[tuple[int, NDArray[np.float64] | BootstrapFailure], None, None]:
    from mixedlm.models.nlmer import _DEFAULT_NLMER_SIMULATE, _NlmerSimulation

    # Draw in the caller to preserve the established sequential random stream.
    # Preparation is local to this bootstrap; result changes are seen next call.
    use_prepared = getattr(result.simulate, "__func__", None) is _DEFAULT_NLMER_SIMULATE
    prepared = None
    for index in range(n_boot):
        response: NDArray[np.float64] | BootstrapFailure
        try:
            if use_prepared:
                if prepared is None:
                    prepared = _NlmerSimulation.prepare(result, include_re=True)
                draw = prepared.draw(rng)
            else:
                draw = result.simulate(nsim=1, seed=rng, use_re=True)
            response = _bootstrap_sample_vector(draw, len(result.y), "Simulated response")
        except Exception as error:
            response = _bootstrap_failure(index, "simulation", error)
        yield index, response


def bootstrap_nlmer(
    result: NlmerResult,
    n_boot: int = 1000,
    seed: RandomSeed = None,
    verbose: bool = False,
    *,
    n_jobs: int = 1,
) -> NlmerBootstrapResult:
    """Parametric bootstrap for nonlinear mixed models.

    Parameters
    ----------
    result : NlmerResult
        A fitted nonlinear mixed model.
    n_boot : int, default 1000
        Positive integer number of bootstrap samples.
    seed : int, RandomState, or Generator, optional
        Local random seed or reusable stream for reproducibility.
    verbose : bool, default False
        Print progress information.
    n_jobs : int, default 1
        Positive refit worker count or -1 for available CPUs, capped at n_boot.
        Parallel workers require a picklable nonlinear model.

    Returns
    -------
    NlmerBootstrapResult
        Bootstrap results containing parameter samples and methods
        for computing confidence intervals and standard errors.

    Notes
    -----
    Simulation or refit exceptions, nonconvergence, nonfinite estimates, and incompatible
    parameter shapes count as failed samples. All components of a failed
    sample remain NaN and are excluded from confidence intervals and
    standard errors. Inspect ``n_failed`` and ``failures`` before interpreting
    the results. Failure records identify the sample row, stage, and reason.
    Confidence bounds and standard errors are NaN when fewer than two samples
    succeed. Residual scales must be positive, and estimates must be real.

    Simulations use the caller's local stream in replicate order, preserving
    integer-seeded results across worker counts. Parallel refits share the
    fitted data once per worker and queue at most two responses per worker.
    Custom model prediction and gradient methods should be deterministic;
    each worker refit receives its own copy of the nonlinear model.

    Examples
    --------
    >>> from mixedlm.nlme.models import SSasymp
    >>> result = nlmer(SSasymp(), data, x_var="time", y_var="conc", group_var="subject")
    >>> boot = bootstrap_nlmer(result, n_boot=500, seed=42)
    >>> boot.ci(level=0.95)
    """
    if isinstance(n_boot, bool | np.bool_) or not isinstance(n_boot, Integral):
        raise TypeError("n_boot must be a positive integer")
    if n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    workers = _bootstrap_worker_count(n_jobs, n_boot)

    n_params = len(result.phi)
    n_theta = len(result.theta)

    phi_samples = np.full((n_boot, n_params), np.nan)
    theta_samples = np.full((n_boot, n_theta), np.nan)
    sigma_samples = np.full(n_boot, np.nan)

    rng = random_stream(seed)
    responses = _nlmer_bootstrap_responses(result, n_boot, rng)
    failures: list[BootstrapFailure] = []

    if n_jobs == 1:
        for b in range(n_boot):
            if verbose and (b + 1) % 100 == 0:
                print(f"Bootstrap iteration {b + 1}/{n_boot}")
            _, response = next(responses)
            sample = _nlmer_bootstrap_refit(result, response, index=b)
            _store_bootstrap_sample(sample, phi_samples, theta_samples, sigma_samples, failures)
    else:
        # Refits only need the numerical model data, not the original frame.
        worker_result = replace(result, _data=None)
        with closing(
            _parallel_bootstrap_tasks(_nlmer_bootstrap_worker, (worker_result,), responses, workers)
        ) as samples:
            for completed, sample in enumerate(samples, start=1):
                if verbose and completed % 100 == 0:
                    print(f"Bootstrap iteration {completed}/{n_boot}")
                _store_bootstrap_sample(sample, phi_samples, theta_samples, sigma_samples, failures)

    return NlmerBootstrapResult(
        n_boot=n_boot,
        phi_samples=phi_samples,
        theta_samples=theta_samples,
        sigma_samples=sigma_samples,
        param_names=list(result.model.param_names),
        original_phi=result.phi.copy(),
        original_theta=result.theta.copy(),
        original_sigma=result.sigma,
        n_failed=len(failures),
        failures=tuple(sorted(failures, key=lambda failure: failure.index)),
    )
