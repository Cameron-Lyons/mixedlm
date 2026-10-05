"""Native group integration preserves likelihoods across worker pool sizes."""

import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import mixedlm
import pytest
from numpy.testing import assert_array_equal

pytest.importorskip("mixedlm._rust")

CASES = list(
    itertools.product(
        ["gaussian", "binomial", "poisson"],
        ["balanced", "uneven", "zeros", "zero_variance"],
        [1, 7, 31, 400],
        [1, 100],
    )
)

# Each process initializes its own pool from RAYON_NUM_THREADS. Pool sizes cannot
# be changed after the extension has initialized the process-global pool.
EVALUATE = r"""
import json
import numpy as np
from scipy import sparse
from mixedlm import _rust
from mixedlm.estimation.laplace import _native_glmm_args
from tests._glmm_oracles import mode_problem

results = {}
for kind in ['gaussian', 'binomial', 'poisson']:
    for layout in ['balanced', 'uneven', 'zeros', 'zero_variance']:
        matrices, family, theta = mode_problem(kind, 'intercept', n_obs=132, n_groups=11)
        args = list(_native_glmm_args(theta, matrices, family))
        if layout == 'uneven':
            groups = np.repeat(np.arange(5), [1, 2, 7, 19, 103])
            design = sparse.csc_matrix((np.ones(132), (np.arange(132), groups)))
            args[2:6] = [design.data, design.indices.astype(np.int64),
                         design.indptr.astype(np.int64), design.shape]
            args[9] = [5]
            theta = np.array([-0.4])
            args[8] = theta
        elif layout == 'zeros':
            args[2][::7] = 0
            args[2][args[4][-2]:] = 0
        elif layout == 'zero_variance':
            theta[:] = 0
        problem = _rust.GlmmProblem(*args[:8], *args[9:])
        for order in [1, 7, 31, 400]:
            for limit in [1, 100]:
                options = dict(maxiter=limit, tol=1e-10)
                raw = _rust.glmm_deviance(*args, order, **options)
                prepared = problem.evaluate(theta, order, **options)
                for value, reference in zip(raw, prepared):
                    np.testing.assert_array_equal(value, reference)
                # A repeated call also checks reuse of the quadrature rule.
                repeated = problem.evaluate(theta, order, **options)
                for value, reference in zip(repeated, prepared):
                    np.testing.assert_array_equal(value, reference)
                key = ':'.join(map(str, [kind, layout, order, limit]))
                results[key] = prepared
print(json.dumps(results))
"""


@pytest.fixture(scope="module")
def worker_results():
    root = Path(__file__).resolve().parents[1]
    python = Path(mixedlm.__file__).resolve().parents[1]
    results = {}
    for workers in [1, 2, 4]:
        env = dict(
            os.environ,
            PYTHONPATH=os.pathsep.join([str(python), str(root)]),
            RAYON_NUM_THREADS=str(workers),
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            NUMEXPR_NUM_THREADS="1",
        )
        run = subprocess.run(
            [sys.executable, "-c", EVALUATE],
            env=env,
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
            timeout=60,
        )
        results[workers] = json.loads(run.stdout)
    return results


@pytest.mark.parametrize("case", CASES)
def test_native_likelihood_and_mode_match_for_all_worker_counts(worker_results, case):
    key = ":".join(map(str, case))
    expected = worker_results[1][key]
    for workers in [2, 4]:
        for actual, reference in zip(worker_results[workers][key], expected, strict=True):
            assert_array_equal(actual, reference)
