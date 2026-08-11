from __future__ import annotations

import numpy as np
from mixedlm.models.lmer_types import PredictResult
from numpy.testing import assert_array_equal


def prediction_result() -> PredictResult:
    return PredictResult(
        fit=np.array([1.0, 2.0, 3.0]),
        se_fit=np.array([0.1, 0.2, 0.3]),
        lower=np.array([0.8, 1.6, 2.4]),
        upper=np.array([1.2, 2.4, 3.6]),
        interval="confidence",
    )


class TestPredictResultArrayProtocol:
    def test_default_conversion_is_zero_copy(self):
        result = prediction_result()

        converted = np.asarray(result)

        assert_array_equal(converted, result.fit)
        assert np.shares_memory(converted, result.fit)

    def test_conversion_honors_dtype(self):
        result = prediction_result()

        converted = np.asarray(result, dtype=np.float32)

        assert converted.dtype == np.float32
        assert_array_equal(converted, result.fit)

    def test_conversion_honors_copy(self):
        result = prediction_result()

        view = np.array(result, copy=False)
        copied = np.array(result, copy=True)

        assert np.shares_memory(view, result.fit)
        assert not np.shares_memory(copied, result.fit)
        assert_array_equal(copied, result.fit)

    def test_supports_scalar_slice_and_advanced_indexing(self):
        result = prediction_result()

        assert result[1] == 2.0
        assert_array_equal(result[1:], [2.0, 3.0])
        assert_array_equal(result[np.array([True, False, True])], [1.0, 3.0])

    def test_exposes_array_metadata_and_iteration(self):
        result = prediction_result()

        assert result.shape == (3,)
        assert result.ndim == 1
        assert result.dtype == np.float64
        assert result.size == 3
        assert list(result) == [1.0, 2.0, 3.0]
