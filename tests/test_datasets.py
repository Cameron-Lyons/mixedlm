from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Callable
from importlib.resources import files

import numpy as np
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

# Independently pinned references derived from the original .rda files at this
# immutable official lme4 commit. Editing the manifest cannot rebaseline them.
_SOURCE_COMMIT = "67d71b0e264bda95f22bdc3ec52261c5fc993d4a"
_CSV_SHA256 = {
    "sleepstudy": "9c69c2a71b48397f9b92941b198ae34b2db14ec8ca5d96a20cd56790e867a2e7",
    "cbpp": "1a2c82ed41e8b4168d943848b61eebfe5d8a79af24e534c6551df0c41e1e17ba",
    "Dyestuff": "4fa4c10f8db617562965d43014e2bb15df244f2cc0a1f83f47b3cd4ac40c17a1",
    "Dyestuff2": "53a2aa3de1b8852e31d3cf1be6168d3c03fa5b6437ac869cac8908c999fd6003",
    "Penicillin": "9a26c574d34d1b6d5a6fcb31d296ffa65fe0339eea4b77f502d90b7f2b4eca24",
    "cake": "51dd01508ef9ca1b11b61d3ee48005bbb2352fc6bcee5677b9d0791c721d029f",
    "Pastes": "0bdc6a81e886faa0d279df29be111bf67adc04713b3255165f65c641f8a65b09",
    "InstEval": "78dbe99f11bc6b9108f2785823cf2ae86aad35314f2f8a0ae3041873782399c7",
    "Arabidopsis": "7d5247d6850e2fbd1bbdf4be7e8ca4a830a861aea28eed9979a252cf7e003e4d",
    "grouseticks": "33599b8d7e0e702f4396d4087409e748e52744383f157c52c71b0537392404ae",
    "VerbAgg": "51f7f5b490c2b62c8f05c11d527b0572d7507973075e8e7978ba2cc2b38b05c6",
}
_SOURCE_SHA256 = {
    "sleepstudy": "65acebb1584c681b0181906744864de8388c7e6ef6052733db7991e18e82a621",
    "cbpp": "c61f75fe66678c55898f0a2313a0636156e5b1a4b2194e154dc9b7601fc80870",
    "Dyestuff": "b243c5c0cd4e911c87c83867fabe7cc7fe580d357926d2fd29a93000241bbc75",
    "Dyestuff2": "1dcd53a310209061c178916e4410477e6af6a5e4571d89381b447e7dd736422f",
    "Penicillin": "263e0257c6ee7747bb1e03f3567aaf6de9ed409fa7c9ad2cbd1b38b2b78064c5",
    "cake": "e21fbee9c8fc1970a9ef2dfe6d701b2aa1370a823914f7630602ae6698d32646",
    "Pastes": "82af78bb57e12e12b3fe103f7c165fcac84563c4cf0702b3f81e27bdb0e2f443",
    "InstEval": "ad7231a53aedc7e4a6dcc5deaec078c8e1845bb8bb3871726030d953797a0f6c",
    "Arabidopsis": "4a507dc49aec01ea32919978737e7b408a4ef84c6ff141dcb57fcfde98b62d5f",
    "grouseticks": "caedfcf7742c3f6de8cf91060de609a044f0cfc7a03876d9ccd968292dbe76bb",
    "VerbAgg": "7e4e01eb8e637c172aca3110b4d3d3cc3b3b7b99a19426bc66534b8680679cd3",
}
_TABLES = {
    "sleepstudy": (load_sleepstudy, 180, ["Reaction", "Days", "Subject"]),
    "cbpp": (load_cbpp, 56, ["herd", "incidence", "size", "period"]),
    "Dyestuff": (load_dyestuff, 30, ["Batch", "Yield"]),
    "Dyestuff2": (load_dyestuff2, 30, ["Batch", "Yield"]),
    "Penicillin": (load_penicillin, 144, ["diameter", "plate", "sample"]),
    "cake": (load_cake, 270, ["replicate", "recipe", "temperature", "angle", "temp"]),
    "Pastes": (load_pastes, 60, ["strength", "batch", "cask", "sample"]),
    "InstEval": (load_insteval, 73421, ["s", "d", "studage", "lectage", "service", "dept", "y"]),
    "Arabidopsis": (
        load_arabidopsis,
        625,
        ["reg", "popu", "gen", "rack", "nutrient", "amd", "status", "total.fruits"],
    ),
    "grouseticks": (
        load_grouseticks,
        403,
        ["INDEX", "TICKS", "BROOD", "HEIGHT", "YEAR", "LOCATION", "cHEIGHT"],
    ),
    "VerbAgg": (
        load_verbagg,
        7584,
        ["Anger", "Gender", "item", "resp", "id", "btype", "situ", "mode", "r2"],
    ),
}


@pytest.mark.parametrize("name", _TABLES)
def test_complete_original_table_matches_independent_source_hash(name: str) -> None:
    loader, rows, columns = _TABLES[name]
    data = loader()
    assert len(data) == rows
    assert list(data)[: len(columns)] == columns
    original = data[columns]
    # Hash every loaded value, including restored precision and original row
    # order. Two documented aliases are excluded from the original R schema.
    payload = original.to_csv(index=False, float_format="%.17g").encode()
    assert hashlib.sha256(payload).hexdigest() == _CSV_SHA256[name]


@pytest.mark.parametrize("name", _TABLES)
def test_bundled_asset_integrity_and_immutable_source_provenance(name: str) -> None:
    resources = files("mixedlm.datasets").joinpath("data")
    provenance = json.loads(resources.joinpath("provenance.json").read_text(encoding="utf-8"))
    assert provenance["source_commit"] == _SOURCE_COMMIT
    assert set(provenance["datasets"]) == set(_TABLES)
    metadata = provenance["datasets"][name]
    assert metadata["source_url"] == (
        f"https://raw.githubusercontent.com/lme4/lme4/{_SOURCE_COMMIT}/data/{name}.rda"
    )
    assert metadata["source_sha256"] == _SOURCE_SHA256[name]
    compressed = resources.joinpath(metadata["file"]).read_bytes()
    assert hashlib.sha256(compressed).hexdigest() == metadata["compressed_sha256"]
    assert metadata["csv_sha256"] == _CSV_SHA256[name]
    assert hashlib.sha256(gzip.decompress(compressed)).hexdigest() == _CSV_SHA256[name]


def test_sleepstudy_preserves_all_subjects_and_the_corrected_reaction_times() -> None:
    data = load_sleepstudy()
    assert data["Subject"].nunique() == 18
    assert data.groupby("Subject")["Days"].apply(list).tolist() == [list(range(10))] * 18
    subject = data.loc[data["Subject"].eq("333") & data["Days"].ge(6)]
    np.testing.assert_array_equal(subject["Reaction"], [332.0265, 348.8399, 333.36, 362.0428])


def test_cbpp_preserves_missing_followup_periods_and_all_fifteen_herds() -> None:
    data = load_cbpp()
    assert set(data["herd"]) == {str(i) for i in range(1, 16)}
    assert data["incidence"].sum() == 99
    assert data.groupby("herd").size().sort_values().tolist() == [1, 3, 4] + [4] * 12
    assert data.iloc[-1].to_dict() == {"herd": "15", "incidence": 0.0, "size": 15.0, "period": "4"}


def test_cake_and_pastes_preserve_the_original_experimental_designs() -> None:
    cake = load_cake()
    np.testing.assert_array_equal(cake["temp"], cake["temperature"].astype(float))
    assert cake.groupby(["recipe", "replicate"]).size().eq(6).all()
    assert cake["angle"].sum() == 8673
    pastes = load_pastes()
    assert pastes["batch"].nunique() == 10
    assert set(pastes["cask"]) == {"a", "b", "c"}
    assert pastes["sample"].nunique() == 30
    assert pastes.groupby("sample").size().eq(2).all()
    np.testing.assert_array_equal(pastes["sample"], pastes["batch"] + ":" + pastes["cask"])


def test_insteval_returns_the_full_original_table_and_factor_labels() -> None:
    data = load_insteval()
    assert data["s"].nunique() == 2972
    assert data["d"].nunique() == 1128
    assert data["dept"].nunique() == 14
    assert set(data["studage"]) == {"2", "4", "6", "8"}
    assert set(data["service"]) == {"0", "1"}
    assert data.iloc[1000]["s"] != data.iloc[0]["s"]
    assert data["y"].sum() == 235369
    assert data.iloc[-1].to_dict() == {
        "s": "2972",
        "d": "2121",
        "studage": "4",
        "lectage": "2",
        "service": "1",
        "dept": "2",
        "y": 3,
    }


def test_count_data_compatibility_aliases_preserve_the_original_columns() -> None:
    arabidopsis = load_arabidopsis()
    assert set(arabidopsis["amd"]) == {"clipped", "unclipped"}
    assert set(arabidopsis["status"]) == {"Normal", "Petri.Plate", "Transplant"}
    assert arabidopsis["total.fruits"].sum() == 18727
    np.testing.assert_array_equal(arabidopsis["total_fruits"], arabidopsis["total.fruits"])
    grouseticks = load_grouseticks()
    assert grouseticks["BROOD"].nunique() == 118
    assert grouseticks["TICKS"].sum() == 2567
    np.testing.assert_array_equal(grouseticks["cTICKS"], grouseticks["TICKS"])
    np.testing.assert_allclose(
        grouseticks["cHEIGHT"],
        grouseticks["HEIGHT"] - grouseticks["HEIGHT"].mean(),
        rtol=0,
        atol=1e-12,
    )


def test_verbagg_preserves_twentyfour_items_and_original_response_labels() -> None:
    data = load_verbagg()
    assert data["id"].nunique() == 316
    assert data["item"].nunique() == 24
    assert data.groupby("id").size().eq(24).all()
    assert set(data["resp"]) == {"no", "perhaps", "yes"}
    assert set(data["r2"]) == {"N", "Y"}
    np.testing.assert_array_equal(data["r2"].eq("Y"), data["resp"].ne("no"))


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
