import warnings

import numpy as np
from mixedlm import condVar, families, glmer, lmer
from mixedlm.estimation.reml import _build_lambda
from mixedlm.matrices.design import RandomEffectStructure
from mixedlm.utils.variance import _conditional_variance_blocks
from scipy import linalg, sparse

from tests._lmer_data import CBPP, SLEEPSTUDY


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
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Model is singular")
        result = lmer(
            "Reaction ~ Days + (Days | Subject)",
            SLEEPSTUDY,
            weights=weights,
        )

    expected = _dense_condvar_reference(result)
    _assert_condvar_equal(condVar(result), expected)

    ranef_condvar = result.ranef(condVar=True).condVar
    expected_variances = {
        group: {term: values for term, values in terms.items() if term != "_cov"}
        for group, terms in expected.items()
    }
    _assert_condvar_equal(ranef_condvar, expected_variances)


def test_weighted_glmer_condvar_matches_dense_reference() -> None:
    result = glmer(
        "y ~ period + (1 | herd)",
        CBPP,
        family=families.Binomial(),
        weights=CBPP["size"].to_numpy(dtype=float),
    )

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
