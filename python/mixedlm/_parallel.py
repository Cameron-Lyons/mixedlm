"""Worker counts and process pools shared by the parallel inference routines."""

from __future__ import annotations

import multiprocessing
import os
import sys
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import contextmanager
from numbers import Integral
from typing import Any, TypeVar

# A forked child inherits the native rayon/faer and BLAS thread pools of a fitted
# parent without their threads and can block forever on its first native solve.
# Like Python 3.14's defaults, prefer forkserver except on macOS, where system
# frameworks are not fork-safe.
_POOL_CONTEXT = multiprocessing.get_context(
    "forkserver"
    if sys.platform != "darwin" and "forkserver" in multiprocessing.get_all_start_methods()
    else "spawn"
)

# Each worker is already one unit of parallelism, so nested BLAS, OpenMP and
# rayon pools would only oversubscribe the cores. BLAS libraries read these when
# they load, so they must be in the environment the worker process starts with.
_WORKER_THREAD_LIMITS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "RAYON_NUM_THREADS",
)
_environment_lock = threading.Lock()

# ProcessPoolExecutor rejects more workers than this on Windows.
_WINDOWS_MAX_WORKERS = 61

_T = TypeVar("_T")


def resolve_n_jobs(n_jobs: int, *, max_tasks: int | None = None) -> int:
    """Return the worker count for ``n_jobs``, where -1 means every available CPU.

    The count is capped at ``max_tasks`` when given, so no worker is left idle.
    """
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, Integral):
        raise TypeError("n_jobs must be -1 or a positive integer")
    if n_jobs != -1 and n_jobs < 1:
        raise ValueError("n_jobs must be -1 or a positive integer")
    workers = (os.cpu_count() or 1) if n_jobs == -1 else int(n_jobs)
    if sys.platform == "win32":
        workers = min(workers, _WINDOWS_MAX_WORKERS)
    if max_tasks is not None:
        workers = min(workers, max(int(max_tasks), 1))
    return workers


@contextmanager
def _worker_thread_limits() -> Iterator[None]:
    with _environment_lock:
        added = [name for name in _WORKER_THREAD_LIMITS if name not in os.environ]
        for name in added:
            os.environ[name] = "1"
        try:
            yield
        finally:
            for name in added:
                os.environ.pop(name, None)


class _ProcessPool(ProcessPoolExecutor):
    def submit(self, fn: Callable[..., _T], /, *args: Any, **kwargs: Any) -> Future[_T]:
        # Non-fork pools start workers on demand inside submit, so the limits are
        # in place exactly when a worker (or the shared forkserver) is launched.
        with _worker_thread_limits():
            return super().submit(fn, *args, **kwargs)


def process_pool(
    max_workers: int,
    initializer: Callable[..., object] | None = None,
    initargs: tuple[Any, ...] = (),
) -> ProcessPoolExecutor:
    """Return a process pool that never forks the calling process.

    Workers start through forkserver or spawn, so callables and arguments must
    be importable and picklable, and scripts need an
    ``if __name__ == "__main__":`` guard. Workers start with one BLAS, OpenMP
    and rayon thread each unless the caller set the corresponding environment
    variable.

    The forkserver is shared by the whole interpreter and its workers inherit
    its environment and imports. Each call replaces the list given to
    ``multiprocessing.set_forkserver_preload``, and a server started here keeps
    the thread limits and the preloaded mixedlm modules for every later
    forkserver pool, including the caller's own. Workers of a server the caller
    started first get neither.
    """
    if _POOL_CONTEXT.get_start_method() == "forkserver":
        # A server started here imports the inference code once; otherwise every
        # worker of every pool imports NumPy, SciPy and mixedlm itself. The
        # default list is ["__main__"], which Python 3.14 imports in the server:
        # a script that fits a model at import time would leave native threads
        # there, and every worker forked from the server could then block.
        _POOL_CONTEXT.set_forkserver_preload(["mixedlm.inference"])
    return _ProcessPool(
        max_workers=max_workers,
        mp_context=_POOL_CONTEXT,
        initializer=initializer,
        initargs=initargs,
    )
