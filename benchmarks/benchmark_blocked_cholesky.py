"""Time repeated native likelihoods with independent and crossed structures.

Run with ``python benchmarks/benchmark_blocked_cholesky.py``. An optional
``--library /path/to/_rust.abi3.so`` allows identical workloads to compare builds.
Preparation and imports are excluded from timings; each call evaluates the same
likelihood parameters. The printed likelihood provides a correctness check when
comparing builds.
"""

import argparse
import importlib.util
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from scipy import sparse


def workload(native, layout):
    rng = np.random.default_rng(751)
    if layout == "independent":
        structures, levels, repeats = 32, 16, 4
        n = structures * levels * repeats
        rows = np.arange(n)
        columns = rows // (levels * repeats) * levels + rows % levels
        z = sparse.csc_matrix((np.ones(n), (rows, columns)), shape=(n, structures * levels))
    else:
        structures, levels, n = 3, 64, 512
        z = sparse.csc_matrix(rng.normal(scale=0.2, size=(n, structures * levels)))
    x = np.column_stack((np.ones(n), rng.normal(size=n)))
    y = x @ np.array([0.4, -0.7]) + rng.normal(size=n)
    design = native.LmmDesign(
        x=x,
        z_data=z.data,
        z_indices=z.indices.astype(np.int64),
        z_indptr=z.indptr.astype(np.int64),
        z_shape=z.shape,
        weights=np.geomspace(0.5, 2.0, n),
        offset=np.zeros(n),
        n_levels=[levels] * structures,
        n_terms=[1] * structures,
        correlated=[False] * structures,
    )
    return design.with_response(y), np.linspace(0.4, 0.9, structures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.rounds < 1 or args.iterations < 1:
        parser.error("rounds and iterations must be positive")
    if args.library:
        spec = importlib.util.spec_from_file_location("_rust", args.library.resolve())
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
    else:
        from mixedlm import _rust as native

    for layout in ["independent", "crossed"]:
        response, theta = workload(native, layout)
        expected = response.deviance(theta, True)
        for operation in ["deviance", "deviance_with_gradient"]:
            evaluate = getattr(response, operation)
            for _ in range(3):
                evaluate(theta, True)
            durations = []
            for _ in range(args.rounds):
                start = perf_counter()
                for _ in range(args.iterations):
                    result = evaluate(theta, True)
                durations.append((perf_counter() - start) / args.iterations)
                value = result[0] if operation.endswith("gradient") else result
                assert np.isfinite(value) and value == expected
            print(
                json.dumps(
                    {
                        "layout": layout,
                        "operation": operation,
                        "median_seconds": median(durations),
                        "deviance": expected,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
