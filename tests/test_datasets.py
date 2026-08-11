from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest
from mixedlm.datasets import (
    load_arabidopsis,
    load_cake,
    load_cbpp,
    load_dyestuff,
    load_dyestuff2,
    load_grouseticks,
    load_insteval,
    load_pastes,
    load_penicillin,
    load_sleepstudy,
    load_verbagg,
)
from mixedlm.datasets.lme4 import _copy_cached_dataset


def test_copy_cache_builds_prototype_once() -> None:
    calls = 0

    def loader() -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return pd.DataFrame({"value": [1.0, 2.0]})

    cached_loader = _copy_cached_dataset(loader)
    first = cached_loader()
    second = cached_loader()

    first.loc[0, "value"] = 99.0
    assert calls == 1
    assert first is not second
    assert second.loc[0, "value"] == 1.0
    assert cached_loader.__name__ == loader.__name__


@pytest.mark.parametrize(
    "loader",
    [
        load_sleepstudy,
        load_cbpp,
        load_dyestuff,
        load_dyestuff2,
        load_penicillin,
        load_cake,
        load_pastes,
        load_insteval,
        load_arabidopsis,
        load_grouseticks,
        load_verbagg,
    ],
)
def test_dataset_loaders_return_independent_frames(
    loader: Callable[[], pd.DataFrame],
) -> None:
    first = loader()
    second = loader()
    numeric_column = first.select_dtypes(include="number").columns[0]
    original = second.loc[second.index[0], numeric_column]

    first.loc[first.index[0], numeric_column] = original + 1

    third = loader()
    assert first is not second
    assert second.loc[second.index[0], numeric_column] == original
    assert third.loc[third.index[0], numeric_column] == original
