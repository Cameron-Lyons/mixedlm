from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

if TYPE_CHECKING:
    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult


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
        alpha = 1 - level

        result: dict[str, tuple[float, float]] = {}

        for i, name in enumerate(self.fixed_names):
            samples = self.beta_samples[:, i]
            samples = samples[~np.isnan(samples)]

            if len(samples) == 0:
                result[name] = (np.nan, np.nan)
                continue

            if method == "percentile":
                lower = np.percentile(samples, 100 * alpha / 2)
                upper = np.percentile(samples, 100 * (1 - alpha / 2))
            elif method == "basic":
                lower = 2 * self.original_beta[i] - np.percentile(samples, 100 * (1 - alpha / 2))
                upper = 2 * self.original_beta[i] - np.percentile(samples, 100 * alpha / 2)
            elif method == "normal":
                se = np.std(samples)
                z = stats.norm.ppf(1 - alpha / 2)
                lower = self.original_beta[i] - z * se
                upper = self.original_beta[i] + z * se
            else:
                raise ValueError(f"Unknown method: {method}")

            result[name] = (float(lower), float(upper))

        return result

    def se(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for i, name in enumerate(self.fixed_names):
            samples = self.beta_samples[:, i]
            samples = samples[~np.isnan(samples)]
            result[name] = float(np.std(samples)) if len(samples) > 0 else np.nan
        return result

    def summary(self) -> str:
        lines = []
        lines.append(f"Parametric bootstrap with {self.n_boot} samples ({self.n_failed} failed)")
        lines.append("")
        lines.append("Fixed effects bootstrap statistics:")
        lines.append("             Original    Mean       Bias     Std.Err")

        for i, name in enumerate(self.fixed_names):
            samples = self.beta_samples[:, i]
            samples = samples[~np.isnan(samples)]
            if len(samples) > 0:
                mean = np.mean(samples)
                bias = mean - self.original_beta[i]
                se = np.std(samples)
                lines.append(
                    f"{name:12} {self.original_beta[i]:10.4f} {mean:10.4f} {bias:10.4f} {se:10.4f}"
                )

        return "\n".join(lines)


def _lmer_bootstrap_worker(
    args: tuple[Any, ...],
) -> tuple[int, NDArray | None, NDArray | None, float | None]:
    import pandas as pd

    from mixedlm.models.lmer import LmerMod

    (
        boot_idx,
        seed,
        formula,
        X,
        Z,
        fixed_names,
        random_structures_data,
        response_name,
        beta,
        theta,
        sigma,
        REML,
    ) = args

    np.random.seed(seed)

    try:
        n = X.shape[0]
        fixed_part = X @ beta

        if Z is not None and Z.shape[1] > 0:
            q = Z.shape[1]
            u_new = np.zeros(q, dtype=np.float64)
            u_idx = 0

            for struct_data in random_structures_data:
                n_levels = struct_data["n_levels"]
                n_terms = struct_data["n_terms"]
                correlated = struct_data["correlated"]
                theta_block = struct_data["theta_block"]

                if correlated:
                    L = np.zeros((n_terms, n_terms))
                    row_indices, col_indices = np.tril_indices(n_terms)
                    L[row_indices, col_indices] = theta_block
                    cov = L @ L.T * sigma**2
                else:
                    cov = np.diag(theta_block**2) * sigma**2

                for g in range(n_levels):
                    b_g = np.random.multivariate_normal(np.zeros(n_terms), cov)
                    for j in range(n_terms):
                        u_new[u_idx + g * n_terms + j] = b_g[j]

                u_idx += n_levels * n_terms

            random_part = Z @ u_new
        else:
            random_part = np.zeros(n)

        noise = np.random.randn(n) * sigma
        y_sim = fixed_part + random_part + noise

        sim_df = pd.DataFrame(X, columns=fixed_names)
        sim_df[response_name] = y_sim

        for struct_data in random_structures_data:
            level_map = struct_data["level_map"]
            grouping_factor = struct_data["grouping_factor"]
            n_terms = struct_data["n_terms"]
            z_slice = struct_data["z_slice"]

            levels = list(level_map.keys())
            z_first_cols = z_slice[:, ::n_terms]
            level_indices = np.argmax(z_first_cols != 0, axis=1)
            group_col = [levels[idx] for idx in level_indices]
            sim_df[grouping_factor] = group_col

        model = LmerMod(formula, sim_df, REML=REML)
        boot_result = model.fit(start=theta)

        return (boot_idx, boot_result.beta.copy(), boot_result.theta.copy(), boot_result.sigma)
    except Exception:
        return (boot_idx, None, None, None)


def _glmer_bootstrap_worker(args: tuple[Any, ...]) -> tuple[int, NDArray | None, NDArray | None]:
    import pandas as pd

    from mixedlm.models.glmer import GlmerMod

    (
        boot_idx,
        seed,
        formula,
        X,
        Z,
        fixed_names,
        random_structures_data,
        response_name,
        beta,
        theta,
        family,
    ) = args

    np.random.seed(seed)

    try:
        n = X.shape[0]

        if Z is not None and Z.shape[1] > 0:
            q = Z.shape[1]
            u_new = np.zeros(q, dtype=np.float64)
            u_idx = 0

            for struct_data in random_structures_data:
                n_levels = struct_data["n_levels"]
                n_terms = struct_data["n_terms"]
                correlated = struct_data["correlated"]
                theta_block = struct_data["theta_block"]

                if correlated:
                    L = np.zeros((n_terms, n_terms))
                    row_indices, col_indices = np.tril_indices(n_terms)
                    L[row_indices, col_indices] = theta_block
                    cov = L @ L.T
                else:
                    cov = np.diag(theta_block**2)

                for g in range(n_levels):
                    b_g = np.random.multivariate_normal(
                        np.zeros(n_terms), cov + 1e-8 * np.eye(n_terms)
                    )
                    for j in range(n_terms):
                        u_new[u_idx + g * n_terms + j] = b_g[j]

                u_idx += n_levels * n_terms

            eta = X @ beta + Z @ u_new
        else:
            eta = X @ beta

        mu = family.link.inverse(eta)
        mu = np.clip(mu, 1e-6, 1 - 1e-6)

        family_name = family.__class__.__name__
        if family_name == "Binomial":
            y_sim = np.random.binomial(1, mu).astype(np.float64)
        elif family_name == "Poisson":
            y_sim = np.random.poisson(mu).astype(np.float64)
        elif family_name == "Gaussian":
            y_sim = np.random.normal(mu, 1.0)
        else:
            y_sim = mu + np.random.randn(n) * 0.1

        sim_df = pd.DataFrame(X, columns=fixed_names)
        sim_df[response_name] = y_sim

        for struct_data in random_structures_data:
            level_map = struct_data["level_map"]
            grouping_factor = struct_data["grouping_factor"]
            n_terms = struct_data["n_terms"]
            z_slice = struct_data["z_slice"]

            levels = list(level_map.keys())
            z_first_cols = z_slice[:, ::n_terms]
            level_indices = np.argmax(z_first_cols != 0, axis=1)
            group_col = [levels[idx] for idx in level_indices]
            sim_df[grouping_factor] = group_col

        model = GlmerMod(formula, sim_df, family=family)
        boot_result = model.fit(start=theta)

        return (boot_idx, boot_result.beta.copy(), boot_result.theta.copy())
    except Exception:
        return (boot_idx, None, None)


def _prepare_lmer_worker_data(result: LmerResult) -> dict[str, Any]:
    n_structs = len(result.matrices.random_structures)
    random_structures_data: list[dict[str, Any]] = [{}] * n_structs
    theta_offset = 0
    z_start = 0

    for s_idx, struct in enumerate(result.matrices.random_structures):
        n_terms = struct.n_terms
        n_theta = n_terms * (n_terms + 1) // 2 if struct.correlated else n_terms
        theta_block = result.theta[theta_offset : theta_offset + n_theta]
        theta_offset += n_theta

        z_end = z_start + struct.n_levels * struct.n_terms
        z_slice = result.matrices.Z[:, z_start:z_end]

        random_structures_data[s_idx] = {
            "n_levels": struct.n_levels,
            "n_terms": struct.n_terms,
            "correlated": struct.correlated,
            "level_map": dict(struct.level_map),
            "grouping_factor": struct.grouping_factor,
            "theta_block": theta_block.copy(),
            "z_slice": z_slice.copy() if hasattr(z_slice, "copy") else np.array(z_slice),
        }
        z_start = z_end

    return {
        "formula": result.formula,
        "X": result.matrices.X.copy(),
        "Z": result.matrices.Z.copy() if result.matrices.Z is not None else None,
        "fixed_names": result.matrices.fixed_names,
        "random_structures_data": random_structures_data,
        "response_name": result.formula.response,
        "beta": result.beta.copy(),
        "theta": result.theta.copy(),
        "sigma": result.sigma,
        "REML": result.REML,
    }


def _prepare_glmer_worker_data(result: GlmerResult) -> dict[str, Any]:
    n_structs = len(result.matrices.random_structures)
    random_structures_data: list[dict[str, Any]] = [{}] * n_structs
    theta_offset = 0
    z_start = 0

    for s_idx, struct in enumerate(result.matrices.random_structures):
        n_terms = struct.n_terms
        n_theta = n_terms * (n_terms + 1) // 2 if struct.correlated else n_terms
        theta_block = result.theta[theta_offset : theta_offset + n_theta]
        theta_offset += n_theta

        z_end = z_start + struct.n_levels * struct.n_terms
        z_slice = result.matrices.Z[:, z_start:z_end]

        random_structures_data[s_idx] = {
            "n_levels": struct.n_levels,
            "n_terms": struct.n_terms,
            "correlated": struct.correlated,
            "level_map": dict(struct.level_map),
            "grouping_factor": struct.grouping_factor,
            "theta_block": theta_block.copy(),
            "z_slice": z_slice.copy() if hasattr(z_slice, "copy") else np.array(z_slice),
        }
        z_start = z_end

    return {
        "formula": result.formula,
        "X": result.matrices.X.copy(),
        "Z": result.matrices.Z.copy() if result.matrices.Z is not None else None,
        "fixed_names": result.matrices.fixed_names,
        "random_structures_data": random_structures_data,
        "response_name": result.formula.response,
        "beta": result.beta.copy(),
        "theta": result.theta.copy(),
        "family": result.family,
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
        import pandas as pd

        from mixedlm.models.lmer import LmerMod

        n_failed = 0
        n = result.matrices.n_obs

        for b in range(n_boot):
            if verbose and (b + 1) % 100 == 0:
                print(f"Bootstrap iteration {b + 1}/{n_boot}")

            np.random.seed(int(seeds[b]))

            try:
                y_sim = _simulate_lmer(result)

                sim_data = result.matrices.X.copy()
                sim_df = pd.DataFrame(sim_data, columns=result.matrices.fixed_names)
                sim_df[result.formula.response] = y_sim

                for struct in result.matrices.random_structures:
                    levels = list(struct.level_map.keys())
                    group_col = []
                    for i in range(n):
                        for lv, idx in struct.level_map.items():
                            if result.matrices.Z[i, idx * struct.n_terms] != 0:
                                group_col.append(lv)
                                break
                        else:
                            group_col.append(levels[0])
                    sim_df[struct.grouping_factor] = group_col

                model = LmerMod(
                    result.formula,
                    sim_df,
                    REML=result.REML,
                )
                boot_result = model.fit(start=result.theta)

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
                worker_data["formula"],
                worker_data["X"],
                worker_data["Z"],
                worker_data["fixed_names"],
                worker_data["random_structures_data"],
                worker_data["response_name"],
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
    n = result.matrices.n_obs
    q = result.matrices.n_random

    fixed_part = result.matrices.X @ result.beta

    if q > 0:
        u_new = np.zeros(q, dtype=np.float64)

        u_idx = 0
        theta_start = 0
        for struct in result.matrices.random_structures:
            n_levels = struct.n_levels
            n_terms = struct.n_terms

            n_theta = n_terms * (n_terms + 1) // 2 if struct.correlated else n_terms

            theta_block = result.theta[theta_start : theta_start + n_theta]

            if struct.correlated:
                L = np.zeros((n_terms, n_terms))
                row_indices, col_indices = np.tril_indices(n_terms)
                L[row_indices, col_indices] = theta_block
                cov = L @ L.T * result.sigma**2
            else:
                cov = np.diag(theta_block**2) * result.sigma**2

            for g in range(n_levels):
                b_g = np.random.multivariate_normal(
                    np.zeros(n_terms),
                    cov,
                )
                for j in range(n_terms):
                    u_new[u_idx + g * n_terms + j] = b_g[j]

            u_idx += n_levels * n_terms
            theta_start += n_theta

        random_part = result.matrices.Z @ u_new
    else:
        random_part = np.zeros(n)

    noise = np.random.randn(n) * result.sigma

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
        import pandas as pd

        from mixedlm.models.glmer import GlmerMod

        n_failed = 0
        n = result.matrices.n_obs

        for b in range(n_boot):
            if verbose and (b + 1) % 100 == 0:
                print(f"Bootstrap iteration {b + 1}/{n_boot}")

            np.random.seed(int(seeds[b]))

            try:
                y_sim = _simulate_glmer(result)

                sim_data = result.matrices.X.copy()
                sim_df = pd.DataFrame(sim_data, columns=result.matrices.fixed_names)
                sim_df[result.formula.response] = y_sim

                for struct in result.matrices.random_structures:
                    levels = list(struct.level_map.keys())
                    group_col = []
                    for i in range(n):
                        for lv, idx in struct.level_map.items():
                            if result.matrices.Z[i, idx * struct.n_terms] != 0:
                                group_col.append(lv)
                                break
                        else:
                            group_col.append(levels[0])
                    sim_df[struct.grouping_factor] = group_col

                model = GlmerMod(
                    result.formula,
                    sim_df,
                    family=result.family,
                )
                boot_result = model.fit(start=result.theta)

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
                worker_data["formula"],
                worker_data["X"],
                worker_data["Z"],
                worker_data["fixed_names"],
                worker_data["random_structures_data"],
                worker_data["response_name"],
                worker_data["beta"],
                worker_data["theta"],
                worker_data["family"],
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
    n = result.matrices.n_obs
    q = result.matrices.n_random

    if q > 0:
        u_new = np.zeros(q, dtype=np.float64)
        u_idx = 0
        theta_start = 0

        for struct in result.matrices.random_structures:
            n_levels = struct.n_levels
            n_terms = struct.n_terms

            n_theta = n_terms * (n_terms + 1) // 2 if struct.correlated else n_terms

            theta_block = result.theta[theta_start : theta_start + n_theta]

            if struct.correlated:
                L = np.zeros((n_terms, n_terms))
                row_indices, col_indices = np.tril_indices(n_terms)
                L[row_indices, col_indices] = theta_block
                cov = L @ L.T
            else:
                cov = np.diag(theta_block**2)

            for g in range(n_levels):
                b_g = np.random.multivariate_normal(
                    np.zeros(n_terms),
                    cov + 1e-8 * np.eye(n_terms),
                )
                for j in range(n_terms):
                    u_new[u_idx + g * n_terms + j] = b_g[j]

            u_idx += n_levels * n_terms
            theta_start += n_theta

        eta = result.matrices.X @ result.beta + result.matrices.Z @ u_new
    else:
        eta = result.matrices.X @ result.beta

    mu = result.family.link.inverse(eta)
    mu = np.clip(mu, 1e-6, 1 - 1e-6)

    family_name = result.family.__class__.__name__

    if family_name == "Binomial":
        y_sim = np.random.binomial(1, mu).astype(np.float64)
    elif family_name == "Poisson":
        y_sim = np.random.poisson(mu).astype(np.float64)
    elif family_name == "Gaussian":
        y_sim = np.random.normal(mu, 1.0)
    else:
        y_sim = mu + np.random.randn(n) * 0.1

    return y_sim
