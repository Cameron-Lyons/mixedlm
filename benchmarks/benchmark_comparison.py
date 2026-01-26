"""
Comprehensive benchmark suite comparing mixedlm performance.

This module benchmarks:
1. mixedlm (Python/Rust) vs lme4 (R) vs MixedModels.jl (Julia)
2. Different REML algorithms (Newton, MM, AI-REML, Riemannian)
3. Scaling with problem size

Requirements:
- mixedlm (this package)
- rpy2 (for R comparisons, optional)
- juliacall (for Julia comparisons, optional)
"""

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class BenchmarkResult:
    name: str
    n_obs: int
    n_groups: int
    n_fixed: int
    n_random: int
    fit_time_ms: float
    algorithm: str
    package: str
    converged: bool
    extra: dict | None = None


def generate_lmm_data(
    n_obs: int,
    n_groups: int,
    n_fixed: int = 2,
    n_random_slopes: int = 0,
    seed: int = 42,
):
    """Generate synthetic data for LMM benchmarking."""
    rng = np.random.default_rng(seed)

    group_labels = rng.integers(0, n_groups, size=n_obs)

    x = rng.standard_normal((n_obs, n_fixed))
    x[:, 0] = 1.0

    beta = rng.standard_normal(n_fixed)

    group_effects = rng.standard_normal(n_groups) * 2.0
    random_intercepts = group_effects[group_labels]

    noise = rng.standard_normal(n_obs)
    y = x @ beta + random_intercepts + noise

    return {
        "y": y,
        "x": x,
        "group": group_labels,
        "n_obs": n_obs,
        "n_groups": n_groups,
        "n_fixed": n_fixed,
    }


def benchmark_mixedlm(data: dict, algorithm: str = "bobyqa") -> BenchmarkResult:
    """Benchmark mixedlm fitting."""
    from mixedlm import lFormula

    try:
        import pandas as pd

        df = pd.DataFrame({
            "y": data["y"],
            "x1": data["x"][:, 1] if data["n_fixed"] > 1 else np.zeros(data["n_obs"]),
            "group": data["group"].astype(str),
        })

        formula = "y ~ x1 + (1 | group)" if data["n_fixed"] > 1 else "y ~ 1 + (1 | group)"

        start = time.perf_counter()
        parsed = lFormula(formula, df)
        from mixedlm.estimation import fit_lmm

        result = fit_lmm(parsed, REML=True, optimizer=algorithm, verbose=0)
        elapsed = time.perf_counter() - start

        return BenchmarkResult(
            name=f"mixedlm_{algorithm}",
            n_obs=data["n_obs"],
            n_groups=data["n_groups"],
            n_fixed=data["n_fixed"],
            n_random=1,
            fit_time_ms=elapsed * 1000,
            algorithm=algorithm,
            package="mixedlm",
            converged=result.converged,
        )
    except Exception as e:
        return BenchmarkResult(
            name=f"mixedlm_{algorithm}",
            n_obs=data["n_obs"],
            n_groups=data["n_groups"],
            n_fixed=data["n_fixed"],
            n_random=1,
            fit_time_ms=float("nan"),
            algorithm=algorithm,
            package="mixedlm",
            converged=False,
            extra={"error": str(e)},
        )


def benchmark_lme4(data: dict) -> BenchmarkResult:
    """Benchmark R lme4 fitting (requires rpy2)."""
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.packages import importr

        numpy2ri.activate()

        lme4 = importr("lme4")

        ro.r.assign("y", data["y"])
        ro.r.assign("x1", data["x"][:, 1] if data["n_fixed"] > 1 else np.zeros(data["n_obs"]))
        ro.r.assign("group", ro.FactorVector(data["group"].astype(str)))

        formula = "y ~ x1 + (1 | group)" if data["n_fixed"] > 1 else "y ~ 1 + (1 | group)"

        start = time.perf_counter()
        ro.r(f"fit <- lmer({formula}, REML=TRUE)")
        elapsed = time.perf_counter() - start

        converged = bool(ro.r("fit@optinfo$conv$lme4$code == 0")[0])

        numpy2ri.deactivate()

        return BenchmarkResult(
            name="lme4",
            n_obs=data["n_obs"],
            n_groups=data["n_groups"],
            n_fixed=data["n_fixed"],
            n_random=1,
            fit_time_ms=elapsed * 1000,
            algorithm="nloptwrap",
            package="lme4",
            converged=converged,
        )
    except ImportError:
        return BenchmarkResult(
            name="lme4",
            n_obs=data["n_obs"],
            n_groups=data["n_groups"],
            n_fixed=data["n_fixed"],
            n_random=1,
            fit_time_ms=float("nan"),
            algorithm="nloptwrap",
            package="lme4",
            converged=False,
            extra={"error": "rpy2 not available"},
        )
    except Exception as e:
        return BenchmarkResult(
            name="lme4",
            n_obs=data["n_obs"],
            n_groups=data["n_groups"],
            n_fixed=data["n_fixed"],
            n_random=1,
            fit_time_ms=float("nan"),
            algorithm="nloptwrap",
            package="lme4",
            converged=False,
            extra={"error": str(e)},
        )


def benchmark_mixedmodels_jl(data: dict) -> BenchmarkResult:
    """Benchmark Julia MixedModels.jl (requires juliacall)."""
    try:
        from juliacall import Main as jl

        jl.seval("using MixedModels, DataFrames")

        jl.y = data["y"]
        jl.x1 = data["x"][:, 1] if data["n_fixed"] > 1 else np.zeros(data["n_obs"])
        jl.group = [str(g) for g in data["group"]]

        jl.seval("df = DataFrame(y=y, x1=x1, group=categorical(group))")

        formula = "y ~ x1 + (1 | group)" if data["n_fixed"] > 1 else "y ~ 1 + (1 | group)"

        start = time.perf_counter()
        jl.seval(f"fit = fit(MixedModel, @formula({formula}), df, REML=true)")
        elapsed = time.perf_counter() - start

        return BenchmarkResult(
            name="MixedModels.jl",
            n_obs=data["n_obs"],
            n_groups=data["n_groups"],
            n_fixed=data["n_fixed"],
            n_random=1,
            fit_time_ms=elapsed * 1000,
            algorithm="default",
            package="MixedModels.jl",
            converged=True,
        )
    except ImportError:
        return BenchmarkResult(
            name="MixedModels.jl",
            n_obs=data["n_obs"],
            n_groups=data["n_groups"],
            n_fixed=data["n_fixed"],
            n_random=1,
            fit_time_ms=float("nan"),
            algorithm="default",
            package="MixedModels.jl",
            converged=False,
            extra={"error": "juliacall not available"},
        )
    except Exception as e:
        return BenchmarkResult(
            name="MixedModels.jl",
            n_obs=data["n_obs"],
            n_groups=data["n_groups"],
            n_fixed=data["n_fixed"],
            n_random=1,
            fit_time_ms=float("nan"),
            algorithm="default",
            package="MixedModels.jl",
            converged=False,
            extra={"error": str(e)},
        )


def run_scaling_benchmark(
    sizes: list[tuple[int, int]] | None = None,
    n_repeats: int = 3,
) -> list[BenchmarkResult]:
    """Run benchmarks across different problem sizes."""
    if sizes is None:
        sizes = [
            (100, 10),
            (500, 25),
            (1000, 50),
            (5000, 100),
            (10000, 200),
            (50000, 500),
        ]

    results = []

    for n_obs, n_groups in sizes:
        print(f"\n{'='*60}")
        print(f"Benchmarking: n_obs={n_obs}, n_groups={n_groups}")
        print("=" * 60)

        data = generate_lmm_data(n_obs, n_groups)

        for _ in range(n_repeats):
            result = benchmark_mixedlm(data, "bobyqa")
            results.append(result)
            if not np.isnan(result.fit_time_ms):
                print(f"  mixedlm (bobyqa): {result.fit_time_ms:.2f}ms")

        result = benchmark_lme4(data)
        results.append(result)
        if not np.isnan(result.fit_time_ms):
            print(f"  lme4: {result.fit_time_ms:.2f}ms")

        result = benchmark_mixedmodels_jl(data)
        results.append(result)
        if not np.isnan(result.fit_time_ms):
            print(f"  MixedModels.jl: {result.fit_time_ms:.2f}ms")

    return results


def run_algorithm_comparison(n_obs: int = 1000, n_groups: int = 50) -> list[BenchmarkResult]:
    """Compare different REML algorithms available in mixedlm."""
    print(f"\n{'='*60}")
    print(f"Algorithm Comparison: n_obs={n_obs}, n_groups={n_groups}")
    print("=" * 60)

    data = generate_lmm_data(n_obs, n_groups)
    results = []

    algorithms = ["bobyqa", "L-BFGS-B", "Nelder-Mead"]

    for algo in algorithms:
        result = benchmark_mixedlm(data, algo)
        results.append(result)
        status = "✓" if result.converged else "✗"
        time_str = f"{result.fit_time_ms:.2f}ms" if not np.isnan(result.fit_time_ms) else "N/A"
        print(f"  {algo:15s}: {time_str:>10s} [{status}]")

    return results


def print_summary(results: list[BenchmarkResult]):
    """Print a summary table of benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    header = f"{'Package':<20} {'Algorithm':<15} {'n_obs':>8} {'n_groups':>8} {'Time (ms)':>12}"
    print(header)
    print("-" * 80)

    for r in results:
        time_str = f"{r.fit_time_ms:.2f}" if not np.isnan(r.fit_time_ms) else "N/A"
        status = "✓" if r.converged else "✗"
        print(f"{r.package:<20} {r.algorithm:<15} {r.n_obs:>8} {r.n_groups:>8} {time_str:>12} {status}")


def save_results(results: list[BenchmarkResult], path: str = "benchmark_results.json"):
    """Save benchmark results to JSON."""
    data = []
    for r in results:
        d = {
            "name": r.name,
            "n_obs": r.n_obs,
            "n_groups": r.n_groups,
            "n_fixed": r.n_fixed,
            "n_random": r.n_random,
            "fit_time_ms": r.fit_time_ms if not np.isnan(r.fit_time_ms) else None,
            "algorithm": r.algorithm,
            "package": r.package,
            "converged": r.converged,
        }
        if r.extra:
            d["extra"] = r.extra
        data.append(d)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {path}")


def main():
    """Run the full benchmark suite."""
    print("=" * 80)
    print("Mixed Effects Models Benchmark Suite")
    print("Comparing: mixedlm (Python/Rust) vs lme4 (R) vs MixedModels.jl (Julia)")
    print("=" * 80)

    all_results = []

    algo_results = run_algorithm_comparison()
    all_results.extend(algo_results)

    scaling_results = run_scaling_benchmark(
        sizes=[(100, 10), (1000, 50), (5000, 100)],
        n_repeats=1,
    )
    all_results.extend(scaling_results)

    print_summary(all_results)
    save_results(all_results)


if __name__ == "__main__":
    main()
