"""
Benchmark the new REML estimation algorithms.

This benchmarks:
1. Min-Max (MM) algorithm - simpler, robust convergence
2. Augmented AI-REML - faster for multi-variance component models
3. Riemannian optimization - handles boundary constraints better
"""

import time

import numpy as np


def generate_multi_random_effect_data(
    n_obs: int = 500,
    n_groups1: int = 20,
    n_groups2: int = 30,
    seed: int = 42,
):
    """Generate data with multiple random effects for testing."""
    rng = np.random.default_rng(seed)

    group1 = rng.integers(0, n_groups1, size=n_obs)
    group2 = rng.integers(0, n_groups2, size=n_obs)

    x = np.ones((n_obs, 2))
    x[:, 1] = rng.standard_normal(n_obs)

    beta = np.array([1.0, 0.5])

    sigma2 = 1.0
    var1 = 2.0
    var2 = 1.5

    re1 = rng.standard_normal(n_groups1) * np.sqrt(var1)
    re2 = rng.standard_normal(n_groups2) * np.sqrt(var2)

    y = x @ beta + re1[group1] + re2[group2] + rng.standard_normal(n_obs) * np.sqrt(sigma2)

    z1 = np.zeros((n_obs, n_groups1))
    z2 = np.zeros((n_obs, n_groups2))
    for i in range(n_obs):
        z1[i, group1[i]] = 1.0
        z2[i, group2[i]] = 1.0

    return {
        "y": y,
        "x": x,
        "z_blocks": [z1, z2],
        "true_variances": [var1, var2],
        "true_sigma2": sigma2,
        "n_obs": n_obs,
    }


def benchmark_mm_reml(data: dict, max_iter: int = 100) -> dict:
    """Benchmark Min-Max REML algorithm."""
    try:
        from mixedlm._rust import mm_reml

        y = data["y"]
        x = data["x"]
        z_blocks = data["z_blocks"]
        init_variances = np.ones(len(z_blocks))
        init_sigma2 = 1.0

        start = time.perf_counter()
        variances, sigma2, iterations, converged = mm_reml(
            y, x, z_blocks, init_variances, init_sigma2, max_iter, 1e-6
        )
        elapsed = time.perf_counter() - start

        return {
            "algorithm": "MM-REML",
            "time_ms": elapsed * 1000,
            "iterations": iterations,
            "converged": converged,
            "variances": variances.tolist(),
            "sigma2": sigma2,
            "error": None,
        }
    except Exception as e:
        return {
            "algorithm": "MM-REML",
            "time_ms": float("nan"),
            "iterations": 0,
            "converged": False,
            "variances": None,
            "sigma2": None,
            "error": str(e),
        }


def benchmark_augmented_ai_reml(data: dict, max_iter: int = 100) -> dict:
    """Benchmark Augmented AI-REML algorithm."""
    try:
        from mixedlm._rust import augmented_ai_reml

        y = data["y"]
        x = data["x"]
        z_blocks = data["z_blocks"]
        init_variances = np.ones(len(z_blocks))
        init_sigma2 = 1.0

        start = time.perf_counter()
        variances, sigma2, iterations, converged = augmented_ai_reml(
            y, x, z_blocks, init_variances, init_sigma2, max_iter, 1e-6
        )
        elapsed = time.perf_counter() - start

        return {
            "algorithm": "Augmented-AI-REML",
            "time_ms": elapsed * 1000,
            "iterations": iterations,
            "converged": converged,
            "variances": variances.tolist(),
            "sigma2": sigma2,
            "error": None,
        }
    except Exception as e:
        return {
            "algorithm": "Augmented-AI-REML",
            "time_ms": float("nan"),
            "iterations": 0,
            "converged": False,
            "variances": None,
            "sigma2": None,
            "error": str(e),
        }


def benchmark_riemannian_reml(data: dict, max_iter: int = 100) -> dict:
    """Benchmark Riemannian REML algorithm."""
    try:
        from mixedlm._rust import riemannian_reml

        y = data["y"]
        x = data["x"]
        z_blocks = data["z_blocks"]
        init_variances = np.ones(len(z_blocks))
        init_sigma2 = 1.0

        start = time.perf_counter()
        variances, sigma2, iterations, converged = riemannian_reml(
            y, x, z_blocks, init_variances, init_sigma2, max_iter, 1e-6, 0.1
        )
        elapsed = time.perf_counter() - start

        return {
            "algorithm": "Riemannian-REML",
            "time_ms": elapsed * 1000,
            "iterations": iterations,
            "converged": converged,
            "variances": variances.tolist(),
            "sigma2": sigma2,
            "error": None,
        }
    except Exception as e:
        return {
            "algorithm": "Riemannian-REML",
            "time_ms": float("nan"),
            "iterations": 0,
            "converged": False,
            "variances": None,
            "sigma2": None,
            "error": str(e),
        }


def run_benchmark(n_obs: int = 500, n_groups1: int = 20, n_groups2: int = 30, n_repeats: int = 5):
    """Run benchmarks for all REML algorithms."""
    print(f"\n{'='*70}")
    print("REML Algorithm Benchmark")
    print(f"Problem size: n={n_obs}, groups1={n_groups1}, groups2={n_groups2}")
    print("=" * 70)

    data = generate_multi_random_effect_data(n_obs, n_groups1, n_groups2)

    print(f"\nTrue variances: {data['true_variances']}")
    print(f"True sigma2: {data['true_sigma2']}")

    algorithms = [
        ("MM-REML", benchmark_mm_reml),
        ("Augmented-AI-REML", benchmark_augmented_ai_reml),
        ("Riemannian-REML", benchmark_riemannian_reml),
    ]

    results = {name: [] for name, _ in algorithms}

    for name, benchmark_fn in algorithms:
        for _ in range(n_repeats):
            result = benchmark_fn(data)
            results[name].append(result)

    print("\n" + "-" * 70)
    print(f"{'Algorithm':<25} {'Time (ms)':<15} {'Iterations':<12} {'Converged':<10}")
    print("-" * 70)

    for name, runs in results.items():
        successful = [r for r in runs if r["converged"]]
        if successful:
            avg_time = np.mean([r["time_ms"] for r in successful])
            avg_iters = np.mean([r["iterations"] for r in successful])
            conv_rate = len(successful) / len(runs) * 100
            print(f"{name:<25} {avg_time:>10.2f}ms   {avg_iters:>8.1f}      {conv_rate:>6.0f}%")

            if successful[0]["variances"]:
                print(f"  -> Estimated variances: {successful[0]['variances']}")
                print(f"  -> Estimated sigma2: {successful[0]['sigma2']:.4f}")
        else:
            error = runs[0].get("error", "Unknown error")
            print(f"{name:<25} {'N/A':>10}      {'N/A':>8}      {'0':>6}%")
            print(f"  -> Error: {error}")

    return results


def run_scaling_benchmark():
    """Run benchmarks at different problem sizes."""
    sizes = [
        (100, 10, 15),
        (500, 25, 35),
        (1000, 50, 75),
        (2000, 100, 150),
    ]

    all_results = {}

    for n_obs, n_groups1, n_groups2 in sizes:
        key = f"{n_obs}_{n_groups1}_{n_groups2}"
        all_results[key] = run_benchmark(n_obs, n_groups1, n_groups2, n_repeats=3)

    print("\n" + "=" * 70)
    print("SCALING SUMMARY")
    print("=" * 70)
    print(f"{'Size':<20} {'MM-REML':<15} {'Aug-AI-REML':<15} {'Riemannian':<15}")
    print("-" * 70)

    for key, results in all_results.items():
        times = []
        for name in ["MM-REML", "Augmented-AI-REML", "Riemannian-REML"]:
            successful = [r for r in results[name] if r["converged"]]
            if successful:
                avg_time = np.mean([r["time_ms"] for r in successful])
                times.append(f"{avg_time:.1f}ms")
            else:
                times.append("N/A")
        print(f"{key:<20} {times[0]:<15} {times[1]:<15} {times[2]:<15}")


def main():
    """Run the full benchmark suite."""
    print("=" * 70)
    print("New REML Algorithms Benchmark Suite")
    print("=" * 70)
    print("\nAlgorithms being tested:")
    print("  1. MM-REML: Min-Max algorithm (simple, robust)")
    print("  2. Augmented-AI-REML: Faster Average Information REML")
    print("  3. Riemannian-REML: Riemannian manifold optimization")

    run_benchmark(n_obs=500, n_groups1=20, n_groups2=30, n_repeats=5)

    run_scaling_benchmark()


if __name__ == "__main__":
    main()
