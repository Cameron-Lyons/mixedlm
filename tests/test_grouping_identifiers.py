"""Quoted grouping identifiers cannot collide with grouping expressions."""

import numpy as np
import pandas as pd
import pytest
from mixedlm import glance, lmer
from mixedlm.formula import parse_formula
from mixedlm.formula.parser import set_cov_type
from mixedlm.matrices import build_model_matrices
from numpy.testing import assert_allclose


def _data():
    rng = np.random.default_rng(21331)
    parent = np.repeat(np.arange(4), 24)
    child = np.tile(np.repeat(np.arange(3), 8), 4)
    literal = np.tile([0, 1, 2, 1], 24)
    joint = 3 * parent + child
    x = rng.normal(size=len(parent))
    y = 1 + 0.4 * x + np.array([-0.6, 0.1, 0.8])[literal]
    y += rng.normal(scale=0.5, size=12)[joint] + rng.normal(scale=0.2, size=len(parent))
    return pd.DataFrame(
        {
            "y": y,
            "x": x,
            "a": parent,
            "b": child,
            "a:b": literal,
            "literal": literal,
            "joint": joint,
        }
    )


# The three-level literal factor is the point of the test, not a modelling choice.
@pytest.mark.filterwarnings("ignore:Grouping factor .* has only 3 levels:UserWarning")
@pytest.mark.parametrize("frame_type", ["pandas", "polars"])
def test_literal_and_joint_factors_remain_distinct_in_fit_prediction_and_reporting(frame_type):
    frame = _data()
    if frame_type == "polars":
        pl = pytest.importorskip("polars")
        frame = pl.from_pandas(frame)
    model = lmer("y ~ x + (1 | `a:b`) + (1 | a:b)", frame)
    manual = lmer("y ~ x + (1 | literal) + (1 | joint)", frame)

    assert model.converged and manual.converged
    assert model.ngrps() == {"`a:b`": 3, "a:b": 12}
    assert_allclose(model.beta, manual.beta, rtol=0, atol=1e-8)
    assert_allclose(model.theta, manual.theta, rtol=0, atol=1e-6)
    assert_allclose(model.predict(frame), manual.predict(frame), rtol=0, atol=1e-7)
    random = model.ranef(condVar=True)
    assert random.values.keys() == {"`a:b`", "a:b"}
    assert random.values["`a:b`"]["(Intercept)"].shape == (3,)
    assert random.values["a:b"]["(Intercept)"].shape == (12,)
    assert random.condVar["`a:b`"]["(Intercept)"].shape == (3,)
    assert random.condVar["a:b"]["(Intercept)"].shape == (12,)
    assert model.VarCorr().groups.keys() == {"`a:b`", "a:b"}
    summary = glance(model).iloc[0]
    assert summary["n_grouping_factors"] == 2
    assert summary["n_groups"] == 15

    assert_allclose(model.refit().predict(frame), model.predict(frame), rtol=0, atol=1e-6)
    assert_allclose(model.update(data=frame).predict(frame), model.predict(frame), atol=1e-6)

    # A novel joint factor must retain the known literal factor's contribution.
    new = _data().iloc[[0]].copy()
    new["b"] = 100
    new["joint"] = 100
    assert_allclose(
        model.predict(new, allow_new_levels=True),
        manual.predict(new, allow_new_levels=True),
        rtol=0,
        atol=1e-7,
    )


def test_covariance_keys_distinguish_literal_colon_from_joint_grouping():
    formula = parse_formula("y ~ x + (x | `a:b`) + (x | a:b)")
    specified = set_cov_type(formula, {"`a:b`": "cs", "a:b": "ar1"})

    assert [term.cov_type for term in specified.random] == ["cs", "ar1"]
    matrices = build_model_matrices(specified, _data())
    assert [
        (structure.grouping_factor, structure.cov_type) for structure in matrices.random_structures
    ] == [
        ("`a:b`", "cs"),
        ("a:b", "ar1"),
    ]


def test_quoted_joint_factor_names_use_unambiguous_formula_spelling():
    data = pd.DataFrame(
        {
            "y": np.arange(6.0),
            "school id": ["a", "a", "b", "b", "c", "c"],
            "class/id": [1, 2, 1, 2, 1, 2],
        }
    )
    formula = parse_formula("y ~ 1 + (1 | `school id`/`class/id`)")
    matrices = build_model_matrices(formula, data)

    assert [structure.grouping_factor for structure in matrices.random_structures] == [
        "`school id`",
        "`school id`:`class/id`",
    ]
