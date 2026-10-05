import numpy as np
import pandas as pd
import pytest
from mixedlm import condVar, families, glmer, lmer
from mixedlm.estimation.reml import _build_lambda
from mixedlm.formula.parser import parse_formula, set_cov_type
from mixedlm.matrices.design import RandomEffectStructure, build_model_matrices
from mixedlm.models.glmer import GlmerResult
from mixedlm.utils.variance import _conditional_variance_blocks
from scipy import linalg, sparse

from tests._datasets import CBPP, CBPP_FORMULA, SLEEPSTUDY


def _blocks_from_full_cov(cond_cov, structures, *, include_cov: bool = True):
    result = {}
    offset = 0
    for struct in structures:
        size = struct.n_levels * struct.n_terms
        structure_cov = cond_cov[offset : offset + size, offset : offset + size]
        variances = np.diag(structure_cov).reshape(struct.n_levels, struct.n_terms)
        group_result = {name: variances[:, index] for index, name in enumerate(struct.term_names)}
        if include_cov and struct.n_terms > 1:
            group_result["_cov"] = np.stack(
                [
                    structure_cov[
                        level * struct.n_terms : (level + 1) * struct.n_terms,
                        level * struct.n_terms : (level + 1) * struct.n_terms,
                    ]
                    for level in range(struct.n_levels)
                ]
            )
        result[struct.grouping_factor] = group_result
        offset += size
    return result


def _dense_condvar_reference(model, *, include_cov: bool = True):
    matrices = model.matrices
    lambda_factor = _build_lambda(model.theta, matrices.random_structures)

    if hasattr(model, "family"):
        mu = model.family.link.inverse(model.linear_predictor())
        clamp_mu = getattr(model.family, "clamp_mu", None)
        mu = clamp_mu(mu) if clamp_mu is not None else np.clip(mu, 1e-10, 1 - 1e-10)
        weights = model.family.weights(mu) * matrices.weights
        scale = 1.0
    else:
        weights = matrices.weights
        scale = model.sigma**2

    weighted_z = matrices.Z.multiply(np.sqrt(np.maximum(weights, 1e-10))[:, np.newaxis])
    precision = lambda_factor.T @ weighted_z.T @ weighted_z @ lambda_factor
    precision = precision.toarray() + np.eye(matrices.n_random)
    lambda_dense = lambda_factor.toarray()
    cond_cov = scale * lambda_dense @ linalg.solve(precision, lambda_dense.T, assume_a="pos")
    return _blocks_from_full_cov(cond_cov, matrices.random_structures, include_cov=include_cov)


def _assert_condvar_equal(actual, expected) -> None:
    assert actual.keys() == expected.keys()
    for group, expected_terms in expected.items():
        assert actual[group].keys() == expected_terms.keys()
        for term, values in expected_terms.items():
            np.testing.assert_allclose(actual[group][term], values, rtol=1e-10, atol=1e-12)


def test_weighted_lmer_condvar_matches_dense_reference() -> None:
    weights = np.linspace(0.25, 2.0, len(SLEEPSTUDY))
    result = lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY, weights=weights)

    expected = _dense_condvar_reference(result)
    _assert_condvar_equal(condVar(result), expected)

    ranef_condvar = result.ranef(condVar=True).condVar
    expected_variances = {
        group: {term: values for term, values in terms.items() if term != "_cov"}
        for group, terms in expected.items()
    }
    _assert_condvar_equal(ranef_condvar, expected_variances)


def test_weighted_glmer_condvar_matches_dense_reference() -> None:
    # Binomial trial counts enter the working weights as prior weights.
    result = glmer(CBPP_FORMULA, CBPP, family=families.Binomial())

    _assert_condvar_equal(condVar(result), _dense_condvar_reference(result))


def test_sparse_blocks_match_dense_reference_for_coupled_structures() -> None:
    rng = np.random.default_rng(42)
    structures = [
        RandomEffectStructure("subject", ["(Intercept)", "x"], 3, 2, True, {}),
        RandomEffectStructure("item", ["(Intercept)"], 4, 1, True, {}),
    ]
    q = 10
    design = rng.normal(size=(q, q))
    precision_dense = design.T @ design + np.eye(q)
    lambda_dense = sparse.block_diag(
        [
            sparse.kron(
                sparse.eye(3),
                np.array([[1.2, 0.0], [0.3, 0.8]]),
            ),
            np.diag(np.linspace(0.7, 1.0, 4)),
        ],
        format="csc",
    ).toarray()
    expected_cov = (
        0.75 * lambda_dense @ linalg.solve(precision_dense, lambda_dense.T, assume_a="pos")
    )

    actual = _conditional_variance_blocks(
        sparse.csc_matrix(precision_dense),
        sparse.csc_matrix(lambda_dense),
        structures,
        scale=0.75,
        include_cov=True,
    )

    _assert_condvar_equal(actual, _blocks_from_full_cov(expected_cov, structures))


@pytest.mark.parametrize("backend", ["dense", "sparse"])
def test_existing_precision_factor_is_reused(backend, monkeypatch) -> None:
    from mixedlm.models import shared_utils

    monkeypatch.setattr(
        shared_utils, "_SPARSE_PROJECTION_MIN_RANDOM", 0 if backend == "sparse" else np.inf
    )
    weights = np.linspace(0.25, 2.0, len(SLEEPSTUDY))
    result = lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY, weights=weights)
    projection = result._weighted_projection
    monkeypatch.setattr(
        shared_utils._RandomEffectFactor,
        "_factorize",
        lambda self: pytest.fail("the existing factorization must be reused"),
    )

    actual = _conditional_variance_blocks(
        projection.random_factor,
        projection.lambda_matrix,
        result.matrices.random_structures,
        scale=result.sigma**2,
        include_cov=True,
    )

    _assert_condvar_equal(actual, _dense_condvar_reference(result))


def test_large_condvar_extraction_never_densifies_full_system(monkeypatch) -> None:
    q = 2048
    diagonal = np.linspace(1.0, 3.0, q)
    precision = sparse.diags(diagonal, format="csc")
    lambda_factor = sparse.eye(q, format="csc")
    structure = RandomEffectStructure(
        grouping_factor="group",
        term_names=["(Intercept)"],
        n_levels=q,
        n_terms=1,
        correlated=True,
        level_map={str(index): index for index in range(q)},
    )
    original_toarray = sparse.csc_matrix.toarray

    def guarded_toarray(matrix, *args, **kwargs):
        if matrix.shape == (q, q):
            raise AssertionError("conditional variance densified the full system")
        return original_toarray(matrix, *args, **kwargs)

    monkeypatch.setattr(sparse.csc_matrix, "toarray", guarded_toarray)

    actual = _conditional_variance_blocks(precision, lambda_factor, [structure])

    np.testing.assert_allclose(actual["group"]["(Intercept)"], 1.0 / diagonal)


def _glmm_result(family, formula, theta, n_groups=6):
    rng = np.random.default_rng(382)
    n = 5 * n_groups
    x = rng.normal(size=n)
    data = pd.DataFrame(
        {"y": np.arange(n) % 2, "x": x, "group": np.arange(n) % n_groups, "item": np.arange(n) % 5}
    )
    matrices = build_model_matrices(
        formula, data, weights=np.geomspace(0.2, 3.0, n), offset=np.linspace(-0.2, 0.2, n)
    )
    return GlmerResult(
        formula=formula,
        matrices=matrices,
        family=family,
        theta=np.asarray(theta, dtype=np.float64),
        beta=np.array([0.3, 0.1]),
        u=rng.normal(scale=0.2, size=matrices.n_random),
        deviance=0.0,
        converged=True,
        n_iter=0,
        nAGQ=1,
    )


@pytest.mark.parametrize("family_type", [families.Binomial, families.Poisson, families.Gamma])
@pytest.mark.parametrize(
    "formula,theta",
    [
        (parse_formula("y ~ x + (x | group)"), [0.8, -0.2, 0.5]),
        (parse_formula("y ~ x + (x | group) + (1 | item)"), [0.8, -0.2, 0.5, 0.4]),
        (parse_formula("y ~ x + (x | group)"), [0.8, -0.2, 0.0]),
        (set_cov_type("y ~ x + (x | group)", "ar1"), [0.8, 0.4]),
    ],
    ids=["slope", "crossed", "boundary", "ar1"],
)
def test_glmm_condvar_needs_no_fixed_effect_projection(
    family_type, formula, theta, monkeypatch
) -> None:
    result = _glmm_result(family_type(), formula, theta)
    expected = _dense_condvar_reference(result)

    def reject_projection(self):
        raise AssertionError("conditional variance should use only the sparse random system")

    monkeypatch.setattr(GlmerResult, "_working_projection", property(reject_projection))

    _assert_condvar_equal(condVar(result), expected)
    expected_variances = {
        group: {term: values for term, values in terms.items() if term != "_cov"}
        for group, terms in expected.items()
    }
    _assert_condvar_equal(result.ranef(condVar=True).condVar, expected_variances)


def test_large_glmm_condvar_never_densifies_full_system(monkeypatch) -> None:
    q = 2048
    result = _glmm_result(
        families.Poisson(), parse_formula("y ~ x + (1 | group)"), [0.8], n_groups=q
    )
    working_weights = result.family.weights(result.fitted(na_expand=False))
    working_weights *= result.matrices.weights
    structure = result.matrices.random_structures[0]
    expected = 0.8**2 / (
        1.0 + 0.8**2 * np.bincount(structure.level_indices, weights=working_weights)
    )

    for matrix_class in (sparse.csc_matrix, sparse.csr_matrix):
        original_toarray = matrix_class.toarray

        def guarded_toarray(matrix, *args, original_toarray=original_toarray, **kwargs):
            if matrix.shape == (q, q):
                raise AssertionError("conditional variance densified the full random system")
            return original_toarray(matrix, *args, **kwargs)

        monkeypatch.setattr(matrix_class, "toarray", guarded_toarray)

    actual = condVar(result)["group"]["(Intercept)"]

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_glmm_diagnostics_reuse_working_weights(monkeypatch) -> None:
    result = _glmm_result(
        families.Binomial(), parse_formula("y ~ x + (x | group)"), [0.8, -0.2, 0.5]
    )
    expected = _dense_condvar_reference(result)
    original_weights = result.family.weights
    calls = 0

    def count_weights(mu):
        nonlocal calls
        calls += 1
        return original_weights(mu)

    monkeypatch.setattr(result.family, "weights", count_weights)

    _assert_condvar_equal(condVar(result), expected)
    assert np.all(np.isfinite(result.vcov()))
    assert np.all(np.isfinite(result.hatvalues()))
    _assert_condvar_equal(condVar(result), expected)
    assert calls == 1


def test_fixed_only_glmm_has_no_conditional_variances() -> None:
    result = _glmm_result(families.Binomial(), parse_formula("y ~ x"), [])
    mu = result.family.link.inverse(result.linear_predictor())
    weights = result.family.weights(mu) * result.matrices.weights
    X = result.matrices.X
    expected_vcov = linalg.inv(X.T @ (weights[:, None] * X))

    assert condVar(result) == {}
    np.testing.assert_allclose(result.vcov(), expected_vcov, rtol=1e-12, atol=1e-12)
