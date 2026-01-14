from __future__ import annotations

import numpy as np
import pytest
from mixedlm.utils.variance import (
    Cv_to_Sv,
    Cv_to_Vv,
    Sv_to_Cv,
    Vv_to_Cv,
    cov2sdcor,
    mlist2vec,
    safe_chol,
    sdcor2cov,
    vec2mlist,
    vec2STlist,
)
from numpy.testing import assert_allclose


class TestSdcor2cov:
    def test_diagonal_case(self):
        sd = np.array([1.0, 2.0, 3.0])
        cov = sdcor2cov(sd)
        expected = np.diag([1.0, 4.0, 9.0])
        assert_allclose(cov, expected)

    def test_with_correlation(self):
        sd = np.array([1.0, 2.0])
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        cov = sdcor2cov(sd, corr)
        expected = np.array([[1.0, 1.0], [1.0, 4.0]])
        assert_allclose(cov, expected)

    def test_identity_correlation(self):
        sd = np.array([2.0, 3.0])
        corr = np.eye(2)
        cov = sdcor2cov(sd, corr)
        expected = np.diag([4.0, 9.0])
        assert_allclose(cov, expected)

    def test_wrong_corr_shape(self):
        sd = np.array([1.0, 2.0])
        corr = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]])
        with pytest.raises(ValueError, match="corr must be"):
            sdcor2cov(sd, corr)


class TestCov2sdcor:
    def test_diagonal_case(self):
        cov = np.diag([1.0, 4.0, 9.0])
        sd, corr = cov2sdcor(cov)
        assert_allclose(sd, [1.0, 2.0, 3.0])
        assert_allclose(corr, np.eye(3))

    def test_with_correlation(self):
        cov = np.array([[1.0, 1.0], [1.0, 4.0]])
        sd, corr = cov2sdcor(cov)
        assert_allclose(sd, [1.0, 2.0])
        assert_allclose(corr, [[1.0, 0.5], [0.5, 1.0]])

    def test_roundtrip(self):
        sd_orig = np.array([1.5, 2.5, 0.8])
        corr_orig = np.array([[1.0, 0.3, -0.2], [0.3, 1.0, 0.4], [-0.2, 0.4, 1.0]])
        cov = sdcor2cov(sd_orig, corr_orig)
        sd, corr = cov2sdcor(cov)
        assert_allclose(sd, sd_orig)
        assert_allclose(corr, corr_orig)


class TestVvToCv:
    def test_2x2_identity(self):
        theta = np.array([1.0, 0.0, 1.0])
        cv = Vv_to_Cv(theta, q=2, sigma=1.0)
        expected = np.array([1.0, 0.0, 1.0])
        assert_allclose(cv, expected)

    def test_with_sigma_scaling(self):
        theta = np.array([1.0, 0.0, 1.0])
        cv = Vv_to_Cv(theta, q=2, sigma=2.0)
        expected = np.array([4.0, 0.0, 4.0])
        assert_allclose(cv, expected)

    def test_wrong_length(self):
        theta = np.array([1.0, 0.0])
        with pytest.raises(ValueError, match="Expected"):
            Vv_to_Cv(theta, q=2)


class TestCvToVv:
    def test_diagonal_case(self):
        cv = np.array([1.0, 0.0, 1.0])
        vv = Cv_to_Vv(cv, q=2, sigma=1.0)
        assert_allclose(vv, [1.0, 0.0, 1.0])

    def test_roundtrip(self):
        theta_orig = np.array([1.5, 0.3, 1.2])
        cv = Vv_to_Cv(theta_orig, q=2, sigma=1.0)
        theta_back = Cv_to_Vv(cv, q=2, sigma=1.0)
        assert_allclose(theta_back, theta_orig, atol=1e-6)

    def test_wrong_length(self):
        cv = np.array([1.0, 0.0])
        with pytest.raises(ValueError, match="Expected"):
            Cv_to_Vv(cv, q=2)


class TestSvToCv:
    def test_2d_case(self):
        sv = np.array([1.0, 2.0, 0.5])
        cv = Sv_to_Cv(sv, q=2)
        expected = np.array([1.0, 1.0, 4.0])
        assert_allclose(cv, expected)

    def test_wrong_length(self):
        sv = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="Expected"):
            Sv_to_Cv(sv, q=2)


class TestCvToSv:
    def test_2d_case(self):
        cv = np.array([1.0, 1.0, 4.0])
        sv = Cv_to_Sv(cv, q=2)
        expected = np.array([1.0, 2.0, 0.5])
        assert_allclose(sv, expected)

    def test_roundtrip(self):
        sv_orig = np.array([1.5, 2.5, 0.4])
        cv = Sv_to_Cv(sv_orig, q=2)
        sv_back = Cv_to_Sv(cv, q=2)
        assert_allclose(sv_back, sv_orig, atol=1e-6)

    def test_wrong_length(self):
        cv = np.array([1.0, 0.0])
        with pytest.raises(ValueError, match="Expected"):
            Cv_to_Sv(cv, q=2)


class TestMlist2vec:
    def test_single_matrix(self):
        m = np.array([[1, 2], [3, 4]])
        vec = mlist2vec([m])
        expected = np.array([1, 3, 2, 4])
        assert_allclose(vec, expected)

    def test_multiple_matrices(self):
        m1 = np.array([[1, 2], [3, 4]])
        m2 = np.array([[5]])
        vec = mlist2vec([m1, m2])
        expected = np.array([1, 3, 2, 4, 5])
        assert_allclose(vec, expected)

    def test_empty_list(self):
        vec = mlist2vec([])
        assert len(vec) == 0


class TestVec2mlist:
    def test_roundtrip(self):
        m1 = np.array([[1, 2], [3, 4]])
        m2 = np.array([[5]])
        vec = mlist2vec([m1, m2])
        matrices = vec2mlist(vec, [(2, 2), (1, 1)])
        assert_allclose(matrices[0], m1)
        assert_allclose(matrices[1], m2)

    def test_single_matrix(self):
        vec = np.array([1, 3, 2, 4])
        matrices = vec2mlist(vec, [(2, 2)])
        expected = np.array([[1, 2], [3, 4]])
        assert_allclose(matrices[0], expected)


class TestVec2STlist:
    def test_single_triangular(self):
        vec = np.array([1.0, 0.5, 2.0])
        matrices = vec2STlist(vec, [2])
        expected = np.array([[1.0, 0.0], [0.5, 2.0]])
        assert_allclose(matrices[0], expected)

    def test_multiple_triangular(self):
        vec = np.array([1.0, 0.5, 2.0, 3.0])
        matrices = vec2STlist(vec, [2, 1])
        assert_allclose(matrices[0], [[1.0, 0.0], [0.5, 2.0]])
        assert_allclose(matrices[1], [[3.0]])


class TestSafeChol:
    def test_well_conditioned(self):
        x = np.array([[4.0, 2.0], [2.0, 5.0]])
        L = safe_chol(x)
        assert_allclose(L @ L.T, x, atol=1e-10)

    def test_near_singular(self):
        x = np.array([[1.0, 0.999], [0.999, 1.0]])
        L = safe_chol(x)
        reconstructed = L @ L.T
        assert_allclose(np.diag(reconstructed), np.diag(x), atol=1e-6)

    def test_positive_definite(self):
        x = np.array([[2.0, 1.0], [1.0, 2.0]])
        L = safe_chol(x)
        assert L.shape == (2, 2)
        assert np.all(np.diag(L) > 0)
