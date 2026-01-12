from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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


def bootstrap_lmer(
    result: LmerResult,
    n_boot: int = 1000,
    seed: int | None = None,
    parallel: bool = False,
    verbose: bool = False,
) -> BootstrapResult:
    import pandas as pd

    from mixedlm.models.lmer import LmerMod

    if seed is not None:
        np.random.seed(seed)

    n = result.matrices.n_obs
    p = result.matrices.n_fixed
    n_theta = len(result.theta)

    beta_samples = np.full((n_boot, p), np.nan)
    theta_samples = np.full((n_boot, n_theta), np.nan)
    sigma_samples = np.full(n_boot, np.nan)

    n_failed = 0

    for b in range(n_boot):
        if verbose and (b + 1) % 100 == 0:
            print(f"Bootstrap iteration {b + 1}/{n_boot}")

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
        for struct in result.matrices.random_structures:
            n_levels = struct.n_levels
            n_terms = struct.n_terms

            theta_start = sum(
                s.n_terms * (s.n_terms + 1) // 2 if s.correlated else s.n_terms
                for s in result.matrices.random_structures[
                    : result.matrices.random_structures.index(struct)
                ]
            )
            n_theta = n_terms * (n_terms + 1) // 2 if struct.correlated else n_terms

            theta_block = result.theta[theta_start : theta_start + n_theta]

            if struct.correlated:
                L = np.zeros((n_terms, n_terms))
                idx = 0
                for i in range(n_terms):
                    for j in range(i + 1):
                        L[i, j] = theta_block[idx]
                        idx += 1
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

        random_part = result.matrices.Z @ u_new
    else:
        random_part = np.zeros(n)

    noise = np.random.randn(n) * result.sigma

    return fixed_part + random_part + noise


def bootstrap_glmer(
    result: GlmerResult,
    n_boot: int = 1000,
    seed: int | None = None,
    type: str = "parametric",
    verbose: bool = False,
) -> BootstrapResult:
    import pandas as pd

    from mixedlm.models.glmer import GlmerMod

    if seed is not None:
        np.random.seed(seed)

    n = result.matrices.n_obs
    p = result.matrices.n_fixed
    n_theta = len(result.theta)

    beta_samples = np.full((n_boot, p), np.nan)
    theta_samples = np.full((n_boot, n_theta), np.nan)

    n_failed = 0

    for b in range(n_boot):
        if verbose and (b + 1) % 100 == 0:
            print(f"Bootstrap iteration {b + 1}/{n_boot}")

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

        for struct in result.matrices.random_structures:
            n_levels = struct.n_levels
            n_terms = struct.n_terms

            theta_start = sum(
                s.n_terms * (s.n_terms + 1) // 2 if s.correlated else s.n_terms
                for s in result.matrices.random_structures[
                    : result.matrices.random_structures.index(struct)
                ]
            )
            n_theta = n_terms * (n_terms + 1) // 2 if struct.correlated else n_terms

            theta_block = result.theta[theta_start : theta_start + n_theta]

            if struct.correlated:
                L = np.zeros((n_terms, n_terms))
                idx = 0
                for i in range(n_terms):
                    for j in range(i + 1):
                        L[i, j] = theta_block[idx]
                        idx += 1
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
