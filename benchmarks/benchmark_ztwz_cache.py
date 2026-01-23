"""Benchmark Z'WZ caching performance improvement."""

import time
import numpy as np
from mixedlm import lFormula, load_sleepstudy, load_insteval
from mixedlm.estimation.reml import LMMOptimizer


def benchmark_dataset(name: str, data, formula: str, n_evals: int = 100):
    """Benchmark a specific dataset with and without caching."""
    print(f"\n{'=' * 70}")
    print(f"Dataset: {name}")
    print(f"Formula: {formula}")

    parsed = lFormula(formula, data)
    print(f"Problem size: n={len(data)}, p={parsed.matrices.n_fixed}, "
          f"q={parsed.matrices.n_random}")

    # Benchmark with Rust backend (includes Z'WZ caching)
    optimizer_rust = LMMOptimizer(
        parsed.matrices, REML=True, verbose=0, use_rust=True
    )
    theta_start = optimizer_rust.get_start_theta()

    start = time.perf_counter()
    for _ in range(n_evals):
        _ = optimizer_rust.objective(theta_start)
    time_rust = time.perf_counter() - start

    # Benchmark with Python backend (no caching)
    optimizer_python = LMMOptimizer(
        parsed.matrices, REML=True, verbose=0, use_rust=False
    )

    start = time.perf_counter()
    for _ in range(n_evals):
        _ = optimizer_python.objective(theta_start)
    time_python = time.perf_counter() - start

    speedup = time_python / time_rust

    print(f"\nResults ({n_evals} evaluations):")
    print(f"  Python (no cache): {time_python:.4f}s ({time_python/n_evals*1000:.4f}ms per eval)")
    print(f"  Rust (with cache): {time_rust:.4f}s ({time_rust/n_evals*1000:.4f}ms per eval)")
    print(f"  Speedup: {speedup:.2f}x")

    return speedup


def main():
    print("Benchmarking Z'WZ Caching Performance")
    print("=" * 70)

    speedups = []

    # Small dataset
    data1 = load_sleepstudy()
    speedup1 = benchmark_dataset(
        "sleepstudy (small)",
        data1,
        "Reaction ~ Days + (1 | Subject)",
        n_evals=200
    )
    speedups.append(("sleepstudy", speedup1))

    # Large dataset
    try:
        data2 = load_insteval()
        speedup2 = benchmark_dataset(
            "InstEval (large)",
            data2,
            "y ~ service + (1 | s) + (1 | d)",
            n_evals=50
        )
        speedups.append(("InstEval", speedup2))
    except Exception as e:
        print(f"\nSkipping InstEval benchmark: {e}")

    print(f"\n{'=' * 70}")
    print("Summary")
    print("=" * 70)
    for name, speedup in speedups:
        print(f"  {name:20s}: {speedup:.2f}x speedup")

    avg_speedup = np.mean([s for _, s in speedups])
    print(f"\n  Average speedup: {avg_speedup:.2f}x")


if __name__ == "__main__":
    main()
