"""Response simulation goes through Family.simulate for every caller."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm import families, glmer
from mixedlm.inference.bootstrap import _simulate_glmer, bootMer
from mixedlm.power import powerSim
from scipy import special

from tests._lmer_data import CBPP


class _VarianceOnlyPoisson(families.CustomFamily):
    """A valid fitting family that defines no response distribution."""

    def __init__(self) -> None:
        super().__init__(link="log")

    def variance(self, mu):
        return mu

    def deviance_resids(self, y, mu, wt):
        return 2 * wt * (special.xlogy(y, y / mu) - y + mu)


class _SubclassedBinomial(families.Binomial):
    pass


COUNTS = pd.DataFrame(
    {
        "y": [0, 2, 1, 4, 3, 5, 2, 6, 1, 3, 4, 7, 9, 8, 11, 1, 0, 2],
        "g": np.repeat(["a", "b", "c", "d", "e", "f"], 3),
    }
)


@pytest.mark.parametrize(
    "family",
    [_VarianceOnlyPoisson(), families.QuasiFamily(families.Poisson(), phi=2.0)],
    ids=["custom", "quasi"],
)
def test_families_without_a_response_distribution_refuse_to_simulate(family) -> None:
    with pytest.raises(NotImplementedError, match="simulate"):
        family.simulate(np.full(5, 7.0), np.random.default_rng(1))


@pytest.mark.parametrize(
    "family",
    [_VarianceOnlyPoisson(), families.QuasiFamily(families.Poisson(), phi=2.0)],
    ids=["custom", "quasi"],
)
def test_simulation_based_inference_raises_for_a_family_without_a_distribution(family) -> None:
    result = glmer("y ~ 1 + (1 | g)", COUNTS, family=family)

    with pytest.raises(NotImplementedError):
        result.simulate(nsim=2, seed=1)
    # One error up front rather than a failure recorded for every replicate.
    with pytest.raises(NotImplementedError):
        bootMer(result, nsim=3, seed=1)
    with pytest.raises(NotImplementedError):
        powerSim(result, test=lambda fitted: True, nsim=2, seed=1)


def test_builtin_families_draw_from_their_weighted_distributions() -> None:
    mu = np.array([0.5, 2.0, 8.0])
    weights = np.array([1.0, 4.0, 0.25])
    trials = np.array([3.0, 10.0, 40.0])
    probabilities = np.array([0.2, 0.5, 0.9])

    def draws(family, mean, **inputs):
        return family.simulate(mean, np.random.default_rng(7), **inputs)

    oracle = np.random.default_rng(7)
    np.testing.assert_array_equal(
        draws(families.Gaussian(), mu, weights=weights), oracle.normal(mu, 1 / np.sqrt(weights))
    )
    oracle = np.random.default_rng(7)
    np.testing.assert_array_equal(
        draws(families.Gamma(), mu, weights=weights), oracle.gamma(weights, mu / weights)
    )
    oracle = np.random.default_rng(7)
    np.testing.assert_array_equal(
        draws(families.InverseGaussian(), mu, weights=weights), oracle.wald(mu, weights)
    )
    oracle = np.random.default_rng(7)
    np.testing.assert_array_equal(
        draws(families.Binomial(), probabilities, trials=trials),
        oracle.binomial(trials.astype(np.int64), probabilities),
    )
    # Bernoulli prior weights are likelihood powers, not trial counts.
    oracle = np.random.default_rng(7)
    np.testing.assert_array_equal(
        draws(families.Binomial(), probabilities, weights=weights),
        oracle.binomial(1, probabilities),
    )


def test_binomial_subclass_keeps_grouped_trial_counts() -> None:
    formula = "incidence / size ~ period + (1 | herd)"
    plain = glmer(formula, CBPP, family=families.Binomial())
    subclassed = glmer(formula, CBPP, family=_SubclassedBinomial())

    expected = plain.simulate(nsim=4, seed=11)
    simulated = subclassed.simulate(nsim=4, seed=11)
    np.testing.assert_array_equal(simulated, expected)
    assert simulated.max() > 1
    assert np.all(simulated <= CBPP["size"].to_numpy()[:, None])

    # The bootstrap refits success proportions of the same draws.
    proportions = _simulate_glmer(subclassed, np.random.RandomState(5))
    np.testing.assert_array_equal(proportions, _simulate_glmer(plain, np.random.RandomState(5)))


def test_custom_family_simulate_receives_weights_and_trials() -> None:
    class RecordingPoisson(families.Poisson):
        def simulate(self, mu, rng=None, *, weights=None, trials=None):
            self.received = (weights, trials)
            return super().simulate(mu, rng, weights=weights, trials=trials)

    family = RecordingPoisson()
    prior = np.arange(1.0, 19.0)
    result = glmer("y ~ 1 + (1 | g)", COUNTS, family=family, weights=prior)
    result.simulate(seed=3)

    weights, trials = family.received
    np.testing.assert_array_equal(weights, prior)
    assert trials is None


def test_unhashable_instance_simulate_hooks_are_supported() -> None:
    class ConstantHook:
        __hash__ = None  # type: ignore[assignment]

        def __call__(self, mu, rng=None):
            return np.full(np.shape(mu), 4.0)

    family = families.Poisson()
    family.simulate = ConstantHook()  # type: ignore[method-assign]
    result = glmer("y ~ 1 + (1 | g)", COUNTS, family=family)

    np.testing.assert_array_equal(result.simulate(seed=3), 4.0)


def test_clip_mu_is_deprecated_in_favor_of_clamp_mu() -> None:
    mu = np.array([-0.5, 0.25, 1.5])
    with pytest.deprecated_call(match="clamp_mu"):
        clipped = families.Binomial().clip_mu(mu, eps=0.01)
    assert clipped is mu
    np.testing.assert_array_equal(mu, [0.01, 0.25, 0.99])
