from __future__ import annotations

import warnings

import numpy as np
from mixedlm.families.base import LogitLink


def test_inverse_logit_handles_extreme_predictors_without_warnings() -> None:
    eta = np.array([-1000.0, 0.0, 1000.0])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        probabilities = LogitLink().inverse(eta)

    np.testing.assert_array_equal(probabilities, np.array([0.0, 0.5, 1.0]))
