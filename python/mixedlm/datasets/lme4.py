"""Original lme4 datasets, bundled for offline use.

The source commit, original R data checksums, and CSV checksums are recorded in
``data/provenance.json``. R factor labels are returned as ordinary object strings
for compatibility with pandas, NumPy, and polars. Numeric values and the original
row and column order are preserved.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import lru_cache, wraps
from importlib.resources import files
from typing import Any

import pandas as pd


def _copy_cached_dataset(
    loader: Callable[[], pd.DataFrame],
) -> Callable[[], pd.DataFrame]:
    """Cache a private dataset prototype and return an independent copy."""

    @lru_cache(maxsize=1)
    def prototype() -> pd.DataFrame:
        return loader()

    @wraps(loader)
    def load_copy() -> pd.DataFrame:
        return prototype().copy(deep=True)

    return load_copy


@lru_cache(maxsize=1)
def _provenance() -> dict[str, Any]:
    resource = files("mixedlm.datasets").joinpath("data").joinpath("provenance.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def _load_original(name: str) -> pd.DataFrame:
    metadata = _provenance()["datasets"][name]
    resource = files("mixedlm.datasets").joinpath("data").joinpath(metadata["file"])
    with resource.open("rb") as stream:
        return pd.read_csv(
            stream,
            compression="gzip",
            dtype=metadata["dtypes"],
            float_precision="round_trip",
        )


@_copy_cached_dataset
def load_sleepstudy() -> pd.DataFrame:
    """Load all 180 original sleepstudy observations from lme4.

    Columns are ``Reaction`` (reaction time in milliseconds), ``Days`` (0–9),
    and ``Subject`` (18 subject identifiers). Days 0–1 are training, day 2 is
    baseline, and sleep restriction begins after day 2.

    Examples
    --------
    >>> from mixedlm import lmer
    >>> model = lmer("Reaction ~ Days + (Days | Subject)", load_sleepstudy())
    """
    return _load_original("sleepstudy")


@_copy_cached_dataset
def load_cbpp() -> pd.DataFrame:
    """Load all 56 original CBPP observations from 15 Ethiopian cattle herds.

    Columns are ``herd``, ``incidence`` (new cases), ``size`` (animals at risk),
    and ``period`` (four time periods). Missing follow-up periods explain why
    some herds have fewer than four rows.

    Examples
    --------
    >>> from mixedlm import glmer, families
    >>> data = load_cbpp()
    >>> model = glmer(
    ...     "incidence / size ~ period + (1 | herd)", data,
    ...     family=families.Binomial()
    ... )
    """
    return _load_original("cbpp")


@_copy_cached_dataset
def load_dyestuff() -> pd.DataFrame:
    """Load 30 original dyestuff yields, five observations in each of six batches.

    Columns are ``Batch`` (A–F) and ``Yield``. A random intercept example is
    ``lmer("Yield ~ 1 + (1 | Batch)", load_dyestuff())``.
    """
    return _load_original("Dyestuff")


@_copy_cached_dataset
def load_dyestuff2() -> pd.DataFrame:
    """Load lme4's 30-row simulated Dyestuff2 table with columns Batch and Yield.

    This original lme4 dataset illustrates boundary variance estimates; the
    returned values are the published table rather than a new simulation.
    """
    return _load_original("Dyestuff2")


@_copy_cached_dataset
def load_penicillin() -> pd.DataFrame:
    """Load 144 original penicillin assays on 24 plates and six samples.

    Columns are ``diameter`` (inhibition zone in millimeters), ``plate`` (a–x),
    and ``sample`` (A–F). Fit crossed effects with
    ``lmer("diameter ~ 1 + (1 | plate) + (1 | sample)", load_penicillin())``.
    """
    return _load_original("Penicillin")


@_copy_cached_dataset
def load_cake() -> pd.DataFrame:
    """Load all 270 original chocolate cake breakage measurements.

    Columns are ``replicate`` (15 levels), ``recipe`` (A–C), ``temperature``
    (six string labels from 175 to 225), ``angle``, and numeric ``temp``.
    Recipes are whole units; temperatures are subunits within replicates.
    Use ``recipe:replicate`` as the grouping factor in the split-plot model.
    """
    return _load_original("cake")


@_copy_cached_dataset
def load_pastes() -> pd.DataFrame:
    """Load 60 original paste assays: two assays per cask in ten batches.

    Columns are ``strength``, ``batch`` (A–J), ``cask`` (a–c within each batch),
    and ``sample`` (30 unique batch/cask identifiers, A:a–J:c).
    ``lmer("strength ~ 1 + (1 | batch/cask)", load_pastes())`` specifies the
    nesting; ``cask`` alone does not identify a sample across batches.
    """
    return _load_original("Pastes")


@_copy_cached_dataset
def load_insteval() -> pd.DataFrame:
    """Load all 73,421 original ETH Zurich instructor evaluation observations.

    Columns are ``s`` (2,972 students), ``d`` (1,128 instructors), ``studage``
    (semester labels 2, 4, 6, 8), ``lectage`` (semester labels 1–6), ``service``
    (0/1 factor labels), ``dept`` (14 departments), and ``y`` (ratings 1–5).
    Select a smaller example explicitly with ``load_insteval().head(1000)``.
    """
    return _load_original("InstEval")


@_copy_cached_dataset
def load_arabidopsis() -> pd.DataFrame:
    """Load all 625 original Arabidopsis clipping/fertilization observations.

    Original columns are ``reg``, ``popu``, ``gen``, ``rack``, ``nutrient``,
    ``amd`` (clipped/unclipped), ``status``, and ``total.fruits``. The supplemental
    ``total_fruits`` column is an identical alias for Python formula usage.
    """
    data = _load_original("Arabidopsis")
    data["total_fruits"] = data["total.fruits"]
    return data


@_copy_cached_dataset
def load_grouseticks() -> pd.DataFrame:
    """Load all 403 original red grouse chick tick observations.

    Original columns are ``INDEX``, ``TICKS``, ``BROOD``, ``HEIGHT``, ``YEAR``,
    ``LOCATION``, and ``cHEIGHT`` (centered height). The supplemental
    ``cTICKS`` column is an identical alias of ``TICKS`` for existing formulas.
    """
    data = _load_original("grouseticks")
    data["cTICKS"] = data["TICKS"]
    return data


@_copy_cached_dataset
def load_verbagg() -> pd.DataFrame:
    """Load all 7,584 original verbal aggression responses from 316 subjects.

    Columns are ``Anger``, ``Gender``, ``item`` (24 items), ``resp``
    (no/perhaps/yes), ``id``, ``btype``, ``situ``, ``mode``, and ``r2`` (N/Y).
    For a binomial model, create a numeric response explicitly:
    ``data["binary"] = data["r2"].eq("Y").astype(int)``.
    """
    return _load_original("VerbAgg")
