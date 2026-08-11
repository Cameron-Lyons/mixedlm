from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

if TYPE_CHECKING:
    from mixedlm.estimation.laplace import GLMMOptimizationResult
    from mixedlm.estimation.reml import OptimizationResult
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
    alpha = 1.0 - level
    lower_percentile = 100.0 * alpha / 2.0
    upper_percentile = 100.0 * (1.0 - alpha / 2.0)
    z_critical = stats.norm.ppf(1.0 - alpha / 2.0) if method == "normal" else None
    result: dict[str, tuple[float, float]] = {}

    for i, name in enumerate(names):
        parameter_samples = _finite_bootstrap_samples(samples, i)
        if len(parameter_samples) == 0:
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
            if len(parameter_samples) < 2:
                result[name] = (np.nan, np.nan)
                continue
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
        )


def _refit_lmer_response(
    matrices: ModelMatrices,
    response: NDArray[np.floating],
    theta: NDArray[np.floating],
    REML: bool,
) -> OptimizationResult:
    """Refit an LMM response against the original validated design matrices."""
    from mixedlm.estimation.reml import LMMOptimizer

    bootstrap_matrices = replace(matrices, y=np.ascontiguousarray(response))
    optimizer = LMMOptimizer(bootstrap_matrices, REML=REML, use_rust=True)
    return optimizer.optimize(start=theta)


def _refit_glmer_response(
    matrices: ModelMatrices,
    response: NDArray[np.floating],
    theta: NDArray[np.floating],
    family: Family,
    nAGQ: int,
) -> GLMMOptimizationResult:
    """Refit a GLMM response against the original validated design matrices."""
    from mixedlm.estimation.laplace import GLMMOptimizer

    bootstrap_matrices = replace(matrices, y=np.ascontiguousarray(response))
    optimizer = GLMMOptimizer(bootstrap_matrices, family, nAGQ=nAGQ)
    return optimizer.optimize(start=theta)


def _lmer_bootstrap_worker(
    args: tuple[Any, ...],
) -> tuple[int, NDArray | None, NDArray | None, float | None]:
    (
        boot_idx,
        seed,
        matrices,
        beta,
        theta,
        sigma,
        REML,
    ) = args

    np.random.seed(seed)

    try:
        y_sim = _simulate_lmer_components(matrices, beta, theta, sigma)
        boot_result = _refit_lmer_response(matrices, y_sim, theta, REML)

        return (boot_idx, boot_result.beta.copy(), boot_result.theta.copy(), boot_result.sigma)
    except Exception:
        return (boot_idx, None, None, None)


def _glmer_bootstrap_worker(args: tuple[Any, ...]) -> tuple[int, NDArray | None, NDArray | None]:
    (
        boot_idx,
        seed,
        matrices,
        beta,
        theta,
        family,
        nAGQ,
    ) = args

    np.random.seed(seed)

    try:
        y_sim = _simulate_glmer_components(matrices, beta, theta, family)
        boot_result = _refit_glmer_response(matrices, y_sim, theta, family, nAGQ)

        return (boot_idx, boot_result.beta.copy(), boot_result.theta.copy())
    except Exception:
        return (boot_idx, None, None)


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
    }


def bootstrap_lmer(
    result: LmerResult,
    n_boot: int = 1000,
    seed: int | None = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> BootstrapResult:
    p = result.matrices.n_fixed
    n_theta = len(result.theta)

    beta_samples = np.full((n_boot, p), np.nan)
    theta_samples = np.full((n_boot, n_theta), np.nan)
    sigma_samples = np.full(n_boot, np.nan)

    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, size=n_boot)

    if n_jobs == 1:
        n_failed = 0

        for b in range(n_boot):
            if verbose and (b + 1) % 100 == 0:
                print(f"Bootstrap iteration {b + 1}/{n_boot}")

            np.random.seed(int(seeds[b]))

            try:
                y_sim = _simulate_lmer(result)
                boot_result = _refit_lmer_response(
                    result.matrices,
                    y_sim,
                    result.theta,
                    result.REML,
                )

                beta_samples[b, :] = boot_result.beta
                theta_samples[b, :] = boot_result.theta
                sigma_samples[b] = boot_result.sigma

            except Exception:
                n_failed += 1
                continue
    else:
        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1

        worker_data = _prepare_lmer_worker_data(result)
        tasks = [
            (
                b,
                int(seeds[b]),
                worker_data["matrices"],
                worker_data["beta"],
                worker_data["theta"],
                worker_data["sigma"],
                worker_data["REML"],
            )
            for b in range(n_boot)
        ]

        n_failed = 0
        completed = 0

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(_lmer_bootstrap_worker, task): task[0] for task in tasks}

            for future in as_completed(futures):
                boot_idx, beta, theta, sigma = future.result()
                completed += 1

                if verbose and completed % 100 == 0:
                    print(f"Bootstrap iteration {completed}/{n_boot}")

                if beta is not None:
                    beta_samples[boot_idx, :] = beta
                    theta_samples[boot_idx, :] = theta
                    sigma_samples[boot_idx] = sigma
                else:
                    n_failed += 1

    return BootstrapResult(
        n_boot=n_boot,
        beta_samples=beta_samples,
        theta_samples=theta_samples,
        sigma_samples=sigma_samples,
        fixed_names=result.matrices.fixed_names,
        original_beta=result.beta,
        original_theta=result.theta,
        original_sigma=result.sigma,
        n_failed=n_failed,
    )


def _simulate_lmer(result: LmerResult) -> NDArray[np.floating]:
    return _simulate_lmer_components(result.matrices, result.beta, result.theta, result.sigma)


def _simulate_lmer_components(
    matrices: ModelMatrices,
    beta: NDArray[np.floating],
    theta: NDArray[np.floating],
    sigma: float,
) -> NDArray[np.floating]:
    n = matrices.n_obs
    q = matrices.n_random

    fixed_part = matrices.X @ beta + matrices.offset

    if q > 0:
        u_new = np.zeros(q, dtype=np.float64)

        u_idx = 0
        theta_start = 0
        for struct in matrices.random_structures:
            n_levels = struct.n_levels
            n_terms = struct.n_terms

            n_theta = n_terms * (n_terms + 1) // 2 if struct.correlated else n_terms

            theta_block = theta[theta_start : theta_start + n_theta]

            if struct.correlated:
                L = np.zeros((n_terms, n_terms))
                row_indices, col_indices = np.tril_indices(n_terms)
                L[row_indices, col_indices] = theta_block
                cov = L @ L.T * sigma**2
            else:
                cov = np.diag(theta_block**2) * sigma**2

            b_all = np.random.multivariate_normal(np.zeros(n_terms), cov, size=n_levels)
            u_new[u_idx : u_idx + n_levels * n_terms] = b_all.ravel()
            u_idx += n_levels * n_terms
            theta_start += n_theta

        random_part = matrices.Z @ u_new
    else:
        random_part = np.zeros(n)

    noise = np.random.randn(n) * sigma / np.sqrt(matrices.weights)

    return fixed_part + random_part + noise


def bootstrap_glmer(
    result: GlmerResult,
    n_boot: int = 1000,
    seed: int | None = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> BootstrapResult:
    p = result.matrices.n_fixed
    n_theta = len(result.theta)

    beta_samples = np.full((n_boot, p), np.nan)
    theta_samples = np.full((n_boot, n_theta), np.nan)

    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, size=n_boot)

    if n_jobs == 1:
        n_failed = 0

        for b in range(n_boot):
            if verbose and (b + 1) % 100 == 0:
                print(f"Bootstrap iteration {b + 1}/{n_boot}")

            np.random.seed(int(seeds[b]))

            try:
                y_sim = _simulate_glmer(result)
                boot_result = _refit_glmer_response(
                    result.matrices,
                    y_sim,
                    result.theta,
                    result.family,
                    result.nAGQ,
                )

                beta_samples[b, :] = boot_result.beta
                theta_samples[b, :] = boot_result.theta

            except Exception:
                n_failed += 1
                continue
    else:
        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1

        worker_data = _prepare_glmer_worker_data(result)
        tasks = [
            (
                b,
                int(seeds[b]),
                worker_data["matrices"],
                worker_data["beta"],
                worker_data["theta"],
                worker_data["family"],
                worker_data["nAGQ"],
            )
            for b in range(n_boot)
        ]

        n_failed = 0
        completed = 0

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(_glmer_bootstrap_worker, task): task[0] for task in tasks}

            for future in as_completed(futures):
                boot_idx, beta, theta = future.result()
                completed += 1

                if verbose and completed % 100 == 0:
                    print(f"Bootstrap iteration {completed}/{n_boot}")

                if beta is not None:
                    beta_samples[boot_idx, :] = beta
                    theta_samples[boot_idx, :] = theta
                else:
                    n_failed += 1

    return BootstrapResult(
        n_boot=n_boot,
        beta_samples=beta_samples,
        theta_samples=theta_samples,
        sigma_samples=None,
        fixed_names=result.matrices.fixed_names,
        original_beta=result.beta,
        original_theta=result.theta,
        original_sigma=None,
        n_failed=n_failed,
    )


def _simulate_glmer(result: GlmerResult) -> NDArray[np.floating]:
    return _simulate_glmer_components(result.matrices, result.beta, result.theta, result.family)


def _simulate_glmer_components(
    matrices: ModelMatrices,
    beta: NDArray[np.floating],
    theta: NDArray[np.floating],
    family: Family,
) -> NDArray[np.floating]:
    q = matrices.n_random

    if q > 0:
        u_new = np.zeros(q, dtype=np.float64)
        u_idx = 0
        theta_start = 0

        for struct in matrices.random_structures:
            n_levels = struct.n_levels
            n_terms = struct.n_terms

            n_theta = n_terms * (n_terms + 1) // 2 if struct.correlated else n_terms

            theta_block = theta[theta_start : theta_start + n_theta]

            if struct.correlated:
                L = np.zeros((n_terms, n_terms))
                row_indices, col_indices = np.tril_indices(n_terms)
                L[row_indices, col_indices] = theta_block
                cov = L @ L.T
            else:
                cov = np.diag(theta_block**2)

            b_all = np.random.multivariate_normal(
                np.zeros(n_terms), cov + 1e-8 * np.eye(n_terms), size=n_levels
            )
            u_new[u_idx : u_idx + n_levels * n_terms] = b_all.ravel()
            u_idx += n_levels * n_terms
            theta_start += n_theta

        eta = matrices.X @ beta + matrices.Z @ u_new + matrices.offset
    else:
        eta = matrices.X @ beta + matrices.offset

    mu = family.link.inverse(eta)
    if family.__class__.__name__ == "Binomial" and matrices.trials is not None:
        mu = family.clamp_mu(mu, eps=1e-6)
        trials = matrices.trials.astype(np.int64)
        successes = np.random.binomial(trials, mu).astype(np.float64)
        return successes / trials
    return family.simulate(mu)


def bootMer(
    model: LmerResult | GlmerResult | NlmerResult,
    nsim: int = 1000,
    seed: int | None = None,
    n_jobs: int = 1,
    verbose: bool = False,
    bootstrap_type: str = "parametric",
) -> BootstrapResult | NlmerBootstrapResult:
    """Model-based (semi-)parametric bootstrap for mixed models.

    This function provides an lme4-compatible interface for bootstrapping
    mixed models. It is a convenience wrapper around bootstrap_lmer and
    bootstrap_glmer that automatically selects the appropriate bootstrap
    function based on the model type.

    Parameters
    ----------
    model : LmerResult or GlmerResult
        A fitted mixed model.
    nsim : int, default 1000
        Number of bootstrap samples.
    seed : int, optional
        Random seed for reproducibility.
    n_jobs : int, default 1
        Number of parallel jobs. Use -1 for all available cores.
    verbose : bool, default False
        Print progress information.
    bootstrap_type : str, default "parametric"
        Type of bootstrap. Currently only "parametric" is supported.
        Parametric bootstrap simulates new responses from the fitted
        model and refits.

    Returns
    -------
    BootstrapResult
        Bootstrap results containing:
        - n_boot: Number of bootstrap samples
        - beta_samples: Fixed effects estimates from each sample
        - theta_samples: Variance parameter estimates from each sample
        - sigma_samples: Residual SD estimates (LMM only)
        - Methods: ci(), se(), summary()

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

    This provides valid inference even when standard errors may be
    unreliable, such as for variance components or in small samples.

    See Also
    --------
    bootstrap_lmer : Bootstrap for linear mixed models.
    bootstrap_glmer : Bootstrap for generalized linear mixed models.
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
        )


def bootstrap_nlmer(
    result: NlmerResult,
    n_boot: int = 1000,
    seed: int | None = None,
    verbose: bool = False,
) -> NlmerBootstrapResult:
    """Parametric bootstrap for nonlinear mixed models.

    Parameters
    ----------
    result : NlmerResult
        A fitted nonlinear mixed model.
    n_boot : int, default 1000
        Number of bootstrap samples.
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool, default False
        Print progress information.

    Returns
    -------
    NlmerBootstrapResult
        Bootstrap results containing parameter samples and methods
        for computing confidence intervals and standard errors.

    Examples
    --------
    >>> from mixedlm.nlme.models import SSasymp
    >>> result = nlmer(SSasymp(), data, x_var="time", y_var="conc", group_var="subject")
    >>> boot = bootstrap_nlmer(result, n_boot=500, seed=42)
    >>> boot.ci(level=0.95)
    """
    n_params = len(result.phi)
    n_theta = len(result.theta)

    phi_samples = np.full((n_boot, n_params), np.nan)
    theta_samples = np.full((n_boot, n_theta), np.nan)
    sigma_samples = np.full(n_boot, np.nan)

    if seed is not None:
        np.random.seed(seed)

    n_failed = 0

    for b in range(n_boot):
        if verbose and (b + 1) % 100 == 0:
            print(f"Bootstrap iteration {b + 1}/{n_boot}")

        try:
            y_sim = result.simulate(nsim=1, use_re=True)
            boot_result = result.refit(y_sim)

            phi_samples[b, :] = boot_result.phi
            theta_samples[b, :] = boot_result.theta
            sigma_samples[b] = boot_result.sigma

        except Exception:
            n_failed += 1
            continue

    return NlmerBootstrapResult(
        n_boot=n_boot,
        phi_samples=phi_samples,
        theta_samples=theta_samples,
        sigma_samples=sigma_samples,
        param_names=list(result.model.param_names),
        original_phi=result.phi.copy(),
        original_theta=result.theta.copy(),
        original_sigma=result.sigma,
        n_failed=n_failed,
    )
