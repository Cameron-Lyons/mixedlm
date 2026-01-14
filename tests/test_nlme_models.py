from __future__ import annotations

import numpy as np
import pytest
from mixedlm.nlme.models import (
    CustomModel,
    SSasymp,
    SSbiexp,
    SSfpl,
    SSgompertz,
    SSlogis,
    SSmicmen,
)
from numpy.testing import assert_allclose


class TestSSasymp:
    def test_properties(self):
        model = SSasymp()
        assert model.name == "SSasymp"
        assert model.param_names == ["Asym", "R0", "lrc"]
        assert model.n_params == 3

    def test_predict(self):
        model = SSasymp()
        params = np.array([10.0, 0.0, 0.0])
        x = np.array([0.0, 1.0, 2.0, 10.0])
        pred = model.predict(params, x)
        assert pred[0] == pytest.approx(0.0, abs=1e-6)
        assert pred[-1] > pred[0]

    def test_gradient_shape(self):
        model = SSasymp()
        params = np.array([10.0, 0.0, 0.0])
        x = np.array([0.0, 1.0, 2.0, 3.0])
        grad = model.gradient(params, x)
        assert grad.shape == (4, 3)

    def test_get_start(self):
        model = SSasymp()
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 5.0, 8.0, 10.0])
        start = model.get_start(x, y)
        assert len(start) == 3


class TestSSlogis:
    def test_properties(self):
        model = SSlogis()
        assert model.name == "SSlogis"
        assert model.param_names == ["Asym", "xmid", "scal"]
        assert model.n_params == 3

    def test_predict_midpoint(self):
        model = SSlogis()
        params = np.array([10.0, 5.0, 1.0])
        x = np.array([5.0])
        pred = model.predict(params, x)
        assert pred[0] == pytest.approx(5.0, abs=0.01)

    def test_predict_limits(self):
        model = SSlogis()
        params = np.array([10.0, 5.0, 1.0])
        x_low = np.array([-100.0])
        x_high = np.array([100.0])
        assert model.predict(params, x_low)[0] < 0.01
        assert model.predict(params, x_high)[0] > 9.99

    def test_gradient_shape(self):
        model = SSlogis()
        params = np.array([10.0, 5.0, 1.0])
        x = np.array([0.0, 5.0, 10.0])
        grad = model.gradient(params, x)
        assert grad.shape == (3, 3)

    def test_get_start(self):
        model = SSlogis()
        x = np.array([0.0, 2.0, 4.0, 6.0])
        y = np.array([0.1, 2.0, 8.0, 9.9])
        start = model.get_start(x, y)
        assert len(start) == 3
        assert start[2] > 0


class TestSSmicmen:
    def test_properties(self):
        model = SSmicmen()
        assert model.name == "SSmicmen"
        assert model.param_names == ["Vm", "K"]
        assert model.n_params == 2

    def test_predict_at_zero(self):
        model = SSmicmen()
        params = np.array([10.0, 2.0])
        x = np.array([0.0])
        pred = model.predict(params, x)
        assert pred[0] == pytest.approx(0.0, abs=1e-10)

    def test_predict_at_K(self):
        model = SSmicmen()
        params = np.array([10.0, 2.0])
        x = np.array([2.0])
        pred = model.predict(params, x)
        assert pred[0] == pytest.approx(5.0, abs=1e-10)

    def test_gradient_shape(self):
        model = SSmicmen()
        params = np.array([10.0, 2.0])
        x = np.array([0.0, 1.0, 2.0, 4.0])
        grad = model.gradient(params, x)
        assert grad.shape == (4, 2)


class TestSSfpl:
    def test_properties(self):
        model = SSfpl()
        assert model.name == "SSfpl"
        assert model.param_names == ["A", "B", "xmid", "scal"]
        assert model.n_params == 4

    def test_predict_limits(self):
        model = SSfpl()
        params = np.array([2.0, 10.0, 5.0, 1.0])
        x_low = np.array([-100.0])
        x_high = np.array([100.0])
        assert model.predict(params, x_low)[0] < 2.01
        assert model.predict(params, x_high)[0] > 9.99

    def test_gradient_shape(self):
        model = SSfpl()
        params = np.array([2.0, 10.0, 5.0, 1.0])
        x = np.array([0.0, 5.0, 10.0])
        grad = model.gradient(params, x)
        assert grad.shape == (3, 4)


class TestSSgompertz:
    def test_properties(self):
        model = SSgompertz()
        assert model.name == "SSgompertz"
        assert model.param_names == ["Asym", "b2", "b3"]
        assert model.n_params == 3

    def test_predict_at_zero(self):
        model = SSgompertz()
        params = np.array([10.0, 2.0, 0.5])
        x = np.array([0.0])
        pred = model.predict(params, x)
        expected = 10.0 * np.exp(-2.0)
        assert pred[0] == pytest.approx(expected, abs=1e-6)

    def test_gradient_shape(self):
        model = SSgompertz()
        params = np.array([10.0, 2.0, 0.5])
        x = np.array([0.0, 1.0, 2.0, 3.0])
        grad = model.gradient(params, x)
        assert grad.shape == (4, 3)


class TestSSbiexp:
    def test_properties(self):
        model = SSbiexp()
        assert model.name == "SSbiexp"
        assert model.param_names == ["A1", "lrc1", "A2", "lrc2"]
        assert model.n_params == 4

    def test_predict_at_zero(self):
        model = SSbiexp()
        params = np.array([5.0, 0.0, 3.0, 0.0])
        x = np.array([0.0])
        pred = model.predict(params, x)
        assert pred[0] == pytest.approx(8.0, abs=1e-6)

    def test_predict_decay(self):
        model = SSbiexp()
        params = np.array([5.0, 0.0, 3.0, -1.0])
        x = np.array([0.0, 1.0, 2.0])
        pred = model.predict(params, x)
        assert pred[0] > pred[1]
        assert pred[1] > pred[2]

    def test_gradient_shape(self):
        model = SSbiexp()
        params = np.array([5.0, 0.0, 3.0, -1.0])
        x = np.array([0.0, 1.0, 2.0])
        grad = model.gradient(params, x)
        assert grad.shape == (3, 4)


class TestCustomModel:
    def test_with_gradient(self):
        def predict_fn(params, x):
            return params[0] * x + params[1]

        def gradient_fn(params, x):
            n = len(x)
            grad = np.zeros((n, 2))
            grad[:, 0] = x
            grad[:, 1] = np.ones(n)
            return grad

        model = CustomModel(
            predict_fn=predict_fn,
            gradient_fn=gradient_fn,
            param_names=["slope", "intercept"],
            name="linear",
        )

        assert model.name == "linear"
        assert model.param_names == ["slope", "intercept"]
        assert model.n_params == 2

        params = np.array([2.0, 1.0])
        x = np.array([0.0, 1.0, 2.0])
        pred = model.predict(params, x)
        assert_allclose(pred, [1.0, 3.0, 5.0])

        grad = model.gradient(params, x)
        assert grad.shape == (3, 2)
        assert_allclose(grad[:, 0], x)
        assert_allclose(grad[:, 1], [1.0, 1.0, 1.0])

    def test_numerical_gradient(self):
        def predict_fn(params, x):
            return params[0] * x**2 + params[1]

        model = CustomModel(
            predict_fn=predict_fn,
            gradient_fn=None,
            param_names=["a", "b"],
            name="quadratic",
        )

        params = np.array([2.0, 1.0])
        x = np.array([0.0, 1.0, 2.0])

        grad = model.gradient(params, x)
        assert grad.shape == (3, 2)
        expected_grad_a = np.array([0.0, 1.0, 4.0])
        assert_allclose(grad[:, 0], expected_grad_a, atol=1e-5)
        assert_allclose(grad[:, 1], [1.0, 1.0, 1.0], atol=1e-5)

    def test_custom_start(self):
        def predict_fn(params, x):
            return params[0] * x

        def start_fn(x, y):
            return np.array([np.mean(y) / np.mean(x)])

        model = CustomModel(
            predict_fn=predict_fn,
            gradient_fn=None,
            param_names=["slope"],
            start_fn=start_fn,
        )

        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])
        start = model.get_start(x, y)
        assert len(start) == 1
        assert start[0] == pytest.approx(2.0, abs=0.01)

    def test_default_start(self):
        def predict_fn(params, x):
            return params[0] * x

        model = CustomModel(
            predict_fn=predict_fn,
            gradient_fn=None,
            param_names=["slope"],
        )

        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])
        start = model.get_start(x, y)
        assert_allclose(start, [1.0])


class TestGradientNumericalCheck:
    def test_ssasymp_gradient(self):
        model = SSasymp()
        params = np.array([10.0, 1.0, 0.5])
        x = np.array([0.5, 1.0, 2.0])

        analytical = model.gradient(params, x)

        eps = 1e-6
        numerical = np.zeros_like(analytical)
        for j in range(3):
            p_plus = params.copy()
            p_plus[j] += eps
            p_minus = params.copy()
            p_minus[j] -= eps
            numerical[:, j] = (model.predict(p_plus, x) - model.predict(p_minus, x)) / (2 * eps)

        assert_allclose(analytical, numerical, atol=1e-4)

    def test_sslogis_gradient(self):
        model = SSlogis()
        params = np.array([10.0, 5.0, 2.0])
        x = np.array([2.0, 5.0, 8.0])

        analytical = model.gradient(params, x)

        eps = 1e-6
        numerical = np.zeros_like(analytical)
        for j in range(3):
            p_plus = params.copy()
            p_plus[j] += eps
            p_minus = params.copy()
            p_minus[j] -= eps
            numerical[:, j] = (model.predict(p_plus, x) - model.predict(p_minus, x)) / (2 * eps)

        assert_allclose(analytical, numerical, atol=1e-4)

    def test_ssmicmen_gradient(self):
        model = SSmicmen()
        params = np.array([10.0, 2.0])
        x = np.array([1.0, 2.0, 4.0])

        analytical = model.gradient(params, x)

        eps = 1e-6
        numerical = np.zeros_like(analytical)
        for j in range(2):
            p_plus = params.copy()
            p_plus[j] += eps
            p_minus = params.copy()
            p_minus[j] -= eps
            numerical[:, j] = (model.predict(p_plus, x) - model.predict(p_minus, x)) / (2 * eps)

        assert_allclose(analytical, numerical, atol=1e-4)
