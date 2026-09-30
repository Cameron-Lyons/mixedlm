from __future__ import annotations

from numbers import Integral
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

RandomStream: TypeAlias = np.random.RandomState | np.random.Generator
RandomSeed: TypeAlias = int | RandomStream | None


def random_stream(seed: RandomSeed, *, legacy: bool = True) -> RandomStream:
    """Keep caller-owned streams or create an isolated stream with legacy seed semantics."""
    if isinstance(seed, (np.random.RandomState, np.random.Generator)):
        return seed
    return (
        np.random.RandomState(seed) if legacy and seed is not None else np.random.default_rng(seed)
    )


def random_seeds(rng: RandomStream, size: int) -> NDArray[np.int64]:
    """Generate deterministic child seeds for serial and parallel bootstrap refits."""
    if isinstance(rng, np.random.Generator):
        return rng.integers(0, 2**31, size=size, dtype=np.int64)
    return rng.randint(0, 2**31, size=size, dtype=np.int64)


def native_seed(seed: RandomSeed, rng: RandomStream) -> int | None:
    """Preserve integer native seeds and advance reusable streams for native batches."""
    if isinstance(seed, (np.random.RandomState, np.random.Generator)):
        return int(random_seeds(rng, 1)[0])
    return None if seed is None else int(seed)


def validate_simulation_count(value: int, name: str = "nsim") -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
