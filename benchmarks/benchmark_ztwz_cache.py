"""Compare native (Rust) and Python LMM objective evaluations.

Both backends prepare the weighted design crossproducts (Z'WZ and related
products) once per optimizer, so the timings compare the per-evaluation solves.
"""

import time

import numpy as np
from mixedlm import lFormula, load_insteval, load_sleepstudy
from mixedlm.estimation.reml import LMMOptimizer


def benchmark_dataset(name: str, data, formula: str, n_evals: int = 100):
    """Time repeated REML objective evaluations on both backends."""
    print(f"\n{'=' * 70}")
    print(f"Dataset: {name}")
    print(f"Formula: {formula}")

    parsed = lFormula(formula, data)
    print(f"Problem size: n={len(data)}, p={parsed.matrices.n_fixed}, q={parsed.matrices.n_random}")

    optimizer_rust = LMMOptimizer(parsed.matrices, REML=True, verbose=0, use_rust=True)
    optimizer_python = LMMOptimizer(parsed.matrices, REML=True, verbose=0, use_rust=False)
    if not optimizer_rust.use_rust:
        raise RuntimeError("the native backend is not available for this model")
    theta_start = optimizer_rust.get_start_theta()

    rust_value = optimizer_rust.objective(theta_start)
    python_value = optimizer_python.objective(theta_start)
    if not np.isclose(rust_value, python_value, rtol=1e-8, atol=1e-8):
        raise AssertionError(f"objectives differ: Rust {rust_value!r}, Python {python_value!r}")

    start = time.perf_counter()
    for _ in range(n_evals):
        optimizer_rust.objective(theta_start)
    time_rust = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(n_evals):
        optimizer_python.objective(theta_start)
    time_python = time.perf_counter() - start

    speedup = time_python / time_rust

    print(f"\nResults ({n_evals} evaluations):")
    print(f"  Python: {time_python:.4f}s ({time_python / n_evals * 1000:.4f}ms per eval)")
    print(f"  Rust: {time_rust:.4f}s ({time_rust / n_evals * 1000:.4f}ms per eval)")
    print(f"  Speedup: {speedup:.2f}x")

    return speedup


def main():
    print("Rust vs Python LMM Objective")
    print("=" * 70)

    speedups = [
        (
            "sleepstudy",
            benchmark_dataset(
                "sleepstudy (small)", load_sleepstudy(), "Reaction ~ Days + (1 | Subject)", 200
            ),
        ),
        (
            "InstEval",
            benchmark_dataset(
                "InstEval (large)", load_insteval(), "y ~ service + (1 | s) + (1 | d)", 50
            ),
        ),
    ]

    print(f"\n{'=' * 70}")
    print("Summary")
    print("=" * 70)
    for name, speedup in speedups:
        print(f"  {name:20s}: {speedup:.2f}x speedup")

    avg_speedup = np.mean([s for _, s in speedups])
    print(f"\n  Average speedup: {avg_speedup:.2f}x")


if __name__ == "__main__":
    main()
