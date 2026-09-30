from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from mixedlm import nlmer
from mixedlm.estimation import nlmm
from mixedlm.inference.bootstrap import bootMer, bootstrap_nlmer
from mixedlm.models.nlmer import NlmerResult
from mixedlm.nlme.models import NonlinearModel, SSasymp
from scipy.optimize import OptimizeResult


class BrokenModel(NonlinearModel):
    def __init__(self, failure=ValueError):
        self.failure = failure

    @property
    def name(self):
        return "broken_model"

    @property
    def param_names(self):
        return ["a"]

    def predict(self, params, x):
        if isinstance(self.failure, type):
            raise self.failure("prediction outside model domain")
        return np.full(len(x), self.failure)

    def gradient(self, params, x):
        return np.ones((len(x), 1))


@pytest.fixture
def data():
    return pd.DataFrame(
        {"x": [1.0, 2.0, 3.0] * 2, "y": [1.0, 2.0, 4.0, 2.0, 3.0, 5.0], "g": [0] * 3 + [1] * 3}
    )


def make_optimizer(data, native=False):
    return nlmm.NLMMOptimizer(
        data["y"].to_numpy(),
        data["x"].to_numpy(),
        data["g"].to_numpy(),
        SSasymp(),
        [0],
        use_rust=native,
    )


def make_result(data, model):
    return NlmerResult(
        model=model,
        group_var="g",
        phi=np.ones(1),
        theta=np.ones(1),
        sigma=1.0,
        b=np.zeros((2, 1)),
        random_params=[0],
        deviance=1.0,
        converged=True,
        n_iter=1,
        x=data["x"].to_numpy(),
        y=data["y"].to_numpy(),
        groups=data["g"].to_numpy(),
        group_levels=["0", "1"],
    )


def stop_at_start(fun, x0, **kwargs):
    theta = np.asarray(x0, dtype=float)
    fun(theta)
    return OptimizeResult(x=theta, success=True, nit=0)


FAILURES = [
    ValueError,
    FloatingPointError,
    OverflowError,
    TypeError,
    np.linalg.LinAlgError,
    np.nan,
    np.inf,
]


@pytest.mark.parametrize("failure", FAILURES)
@pytest.mark.parametrize("entry", ["fit", "refit"])
def test_model_failures_never_return_a_converged_fit(data, failure, entry):
    model = BrokenModel(failure)
    with pytest.raises(RuntimeError, match="did not produce a valid fit"):
        if entry == "fit":
            nlmer(model, data, x_var="x", y_var="y", group_var="g")
        else:
            make_result(data, model).refit()


def test_failure_message_identifies_the_model_error(data):
    with pytest.raises(RuntimeError, match="ValueError: prediction outside model domain"):
        nlmer(BrokenModel(), data, x_var="x", y_var="y", group_var="g")


@pytest.mark.parametrize("entry", ["bootstrap_nlmer", "bootMer"])
def test_invalid_refits_are_recorded_as_bootstrap_failures(data, entry):
    result = make_result(data, BrokenModel())
    with patch.object(result, "simulate", return_value=result.y):
        boot = (
            bootstrap_nlmer(result, n_boot=2, seed=42)
            if entry == "bootstrap_nlmer"
            else bootMer(result, nsim=2, seed=42)
        )
    assert boot.n_failed == 2
    assert np.isnan(boot.phi_samples).all()
    assert np.isnan(boot.theta_samples).all()
    assert np.isnan(boot.sigma_samples).all()
    assert np.isnan(boot.ci()["a"]).all()


INVALID_EVALUATIONS = [
    (0, np.nan, "deviance"),
    (0, np.inf, "deviance"),
    (0, -np.inf, "deviance"),
    (0, 1e100, "failure penalty"),
    (0, np.array([1.0]), "deviance has shape"),
    (0, 1 + 1j, "deviance"),
    (1, np.array([np.nan, 1.0, 2.0]), "fixed parameters"),
    (1, np.array([1.0, np.inf, 2.0]), "fixed parameters"),
    (1, np.ones(2), "fixed parameters has shape"),
    (1, np.ones((1, 3)), "fixed parameters has shape"),
    (1, np.array([1.0, 2.0, 1j]), "fixed parameters"),
    (2, np.array([[np.nan], [1.0]]), "random effects"),
    (2, np.array([[1.0], [np.inf]]), "random effects"),
    (2, np.ones(2), "random effects has shape"),
    (2, np.ones((1, 2)), "random effects has shape"),
    (2, np.ones((3, 1)), "random effects has shape"),
    (2, np.array([[1j], [1.0]]), "random effects"),
    (3, np.nan, "residual scale"),
    (3, np.inf, "residual scale"),
    (3, -np.inf, "residual scale"),
    (3, 0.0, "strictly positive"),
    (3, -1.0, "strictly positive"),
    (3, np.array([1.0]), "residual scale has shape"),
    (3, 1j, "residual scale"),
]


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("component,value,message", INVALID_EVALUATIONS)
def test_invalid_final_evaluations_raise_even_after_optimizer_success(
    data, native, component, value, message
):
    optimizer = make_optimizer(data, native)
    # Exercise both dispatch paths independently of extension availability.
    optimizer.use_rust = native
    evaluation = [-12.0, np.array([5.0, 1.0, -0.5]), np.zeros((2, 1)), 1.0]
    evaluation[component] = value
    target = "_nlmm_deviance_rust" if native else "nlmm_deviance"
    with (
        patch.object(nlmm, target, return_value=tuple(evaluation)) as backend,
        patch.object(nlmm, "minimize", side_effect=stop_at_start),
        pytest.raises(RuntimeError, match=message),
    ):
        optimizer.optimize()
    assert backend.call_count == 1  # The failed final point remains cached.


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_invalid_final_variance_parameters_raise(data, value):
    optimizer = make_optimizer(data)

    def invalid_final(fun, x0, **kwargs):
        theta = np.array([value])
        assert np.isfinite(fun(theta))
        return OptimizeResult(x=theta, success=True, nit=0)

    with (
        patch.object(nlmm, "minimize", side_effect=invalid_final),
        pytest.raises(RuntimeError, match="variance parameters must be finite"),
    ):
        optimizer.optimize()


@pytest.mark.parametrize("native", [False, True])
def test_failed_trial_can_be_followed_by_a_valid_fit(data, native):
    optimizer = make_optimizer(data, native)
    optimizer.use_rust = native
    target = "_nlmm_deviance_rust" if native else "nlmm_deviance"
    good = (-12.0, np.array([5.0, 1.0, -0.5]), np.zeros((2, 1)), 1.0)

    def evaluate(theta, *args, **kwargs):
        if theta[0] < 0.9:
            raise ValueError("outside trial domain")
        return good

    with patch.object(nlmm, target, side_effect=evaluate) as backend:
        penalty = optimizer.objective(np.array([0.5]))
        assert np.isfinite(penalty)
        assert penalty > 1e50
        assert optimizer.objective(np.array([0.5])) == penalty
        assert backend.call_count == 1
        assert optimizer.objective(np.array([1.0])) == -12.0
        with pytest.raises(RuntimeError, match="outside trial domain"):
            optimizer.optimize(start_theta=np.array([0.5]))
        result = optimizer.optimize(start_theta=np.array([1.0]))
    assert result.converged
    assert result.deviance == -12.0


@pytest.mark.parametrize("success", [False, True])
@pytest.mark.parametrize(
    "deviance", [-1e101, -12.0, 0.0, np.nextafter(1e100, 0), np.nextafter(1e100, np.inf)]
)
def test_finite_valid_fits_keep_optimizer_status(data, success, deviance):
    optimizer = make_optimizer(data)
    evaluation = (deviance, np.array([5.0, 1.0, -0.5]), np.zeros((2, 1)), 1.0)

    def finished(fun, x0, **kwargs):
        theta = np.asarray(x0)
        fun(theta)
        return OptimizeResult(x=theta, success=success, nit=7)

    with (
        patch.object(nlmm, "nlmm_deviance", return_value=evaluation),
        patch.object(nlmm, "minimize", side_effect=finished),
    ):
        result = optimizer.optimize()
    assert result.converged is success
    assert result.deviance == deviance
    assert result.n_iter == 7
    np.testing.assert_array_equal(result.phi, evaluation[1])
