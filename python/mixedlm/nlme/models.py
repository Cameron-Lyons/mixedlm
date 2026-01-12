from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class NonlinearModel(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def param_names(self) -> list[str]:
        pass

    @property
    def n_params(self) -> int:
        return len(self.param_names)

    @abstractmethod
    def predict(
        self, params: NDArray[np.floating], x: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        pass

    @abstractmethod
    def gradient(
        self, params: NDArray[np.floating], x: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        pass

    def get_start(self, x: NDArray[np.floating], y: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.ones(self.n_params, dtype=np.float64)


@dataclass
class SSasymp(NonlinearModel):
    @property
    def name(self) -> str:
        return "SSasymp"

    @property
    def param_names(self) -> list[str]:
        return ["Asym", "R0", "lrc"]

    def predict(
        self, params: NDArray[np.floating], x: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        Asym, R0, lrc = params[0], params[1], params[2]
        return Asym + (R0 - Asym) * np.exp(-np.exp(lrc) * x)

    def gradient(
        self, params: NDArray[np.floating], x: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        Asym, R0, lrc = params[0], params[1], params[2]
        rc = np.exp(lrc)
        exp_term = np.exp(-rc * x)

        grad = np.zeros((len(x), 3), dtype=np.float64)
        grad[:, 0] = 1 - exp_term
        grad[:, 1] = exp_term
        grad[:, 2] = -(R0 - Asym) * rc * x * exp_term

        return grad

    def get_start(self, x: NDArray[np.floating], y: NDArray[np.floating]) -> NDArray[np.floating]:
        Asym = np.max(y)
        R0 = np.min(y)
        lrc = np.log(1.0 / np.max(x)) if np.max(x) > 0 else 0.0
        return np.array([Asym, R0, lrc], dtype=np.float64)


@dataclass
class SSlogis(NonlinearModel):
    @property
    def name(self) -> str:
        return "SSlogis"

    @property
    def param_names(self) -> list[str]:
        return ["Asym", "xmid", "scal"]

    def predict(
        self, params: NDArray[np.floating], x: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        Asym, xmid, scal = params[0], params[1], params[2]
        return Asym / (1 + np.exp((xmid - x) / scal))

    def gradient(
        self, params: NDArray[np.floating], x: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        Asym, xmid, scal = params[0], params[1], params[2]
        exp_term = np.exp((xmid - x) / scal)
        denom = 1 + exp_term
        denom_sq = denom**2

        grad = np.zeros((len(x), 3), dtype=np.float64)
        grad[:, 0] = 1 / denom
        grad[:, 1] = -Asym * exp_term / (scal * denom_sq)
        grad[:, 2] = Asym * (xmid - x) * exp_term / (scal**2 * denom_sq)

        return grad

    def get_start(self, x: NDArray[np.floating], y: NDArray[np.floating]) -> NDArray[np.floating]:
        Asym = np.max(y)
        xmid = np.median(x)
        scal = (np.max(x) - np.min(x)) / 4
        if scal <= 0:
            scal = 1.0
        return np.array([Asym, xmid, scal], dtype=np.float64)


@dataclass
class SSmicmen(NonlinearModel):
    @property
    def name(self) -> str:
        return "SSmicmen"

    @property
    def param_names(self) -> list[str]:
        return ["Vm", "K"]

    def predict(
        self, params: NDArray[np.floating], x: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        Vm, K = params[0], params[1]
        return Vm * x / (K + x)

    def gradient(
        self, params: NDArray[np.floating], x: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        Vm, K = params[0], params[1]
        denom = K + x
        denom_sq = denom**2

        grad = np.zeros((len(x), 2), dtype=np.float64)
        grad[:, 0] = x / denom
        grad[:, 1] = -Vm * x / denom_sq

        return grad

    def get_start(self, x: NDArray[np.floating], y: NDArray[np.floating]) -> NDArray[np.floating]:
        Vm = np.max(y) * 1.1
        K = np.median(x)
        return np.array([Vm, K], dtype=np.float64)


@dataclass
class SSfpl(NonlinearModel):
    @property
    def name(self) -> str:
        return "SSfpl"

    @property
    def param_names(self) -> list[str]:
        return ["A", "B", "xmid", "scal"]

    def predict(
        self, params: NDArray[np.floating], x: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        A, B, xmid, scal = params[0], params[1], params[2], params[3]
        return A + (B - A) / (1 + np.exp((xmid - x) / scal))

    def gradient(
        self, params: NDArray[np.floating], x: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        A, B, xmid, scal = params[0], params[1], params[2], params[3]
        exp_term = np.exp((xmid - x) / scal)
        denom = 1 + exp_term
        denom_sq = denom**2

        grad = np.zeros((len(x), 4), dtype=np.float64)
        grad[:, 0] = 1 - 1 / denom
        grad[:, 1] = 1 / denom
        grad[:, 2] = -(B - A) * exp_term / (scal * denom_sq)
        grad[:, 3] = (B - A) * (xmid - x) * exp_term / (scal**2 * denom_sq)

        return grad

    def get_start(self, x: NDArray[np.floating], y: NDArray[np.floating]) -> NDArray[np.floating]:
        A = np.min(y)
        B = np.max(y)
        xmid = np.median(x)
        scal = (np.max(x) - np.min(x)) / 4
        if scal <= 0:
            scal = 1.0
        return np.array([A, B, xmid, scal], dtype=np.float64)


@dataclass
class SSgompertz(NonlinearModel):
    @property
    def name(self) -> str:
        return "SSgompertz"

    @property
    def param_names(self) -> list[str]:
        return ["Asym", "b2", "b3"]

    def predict(
        self, params: NDArray[np.floating], x: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        Asym, b2, b3 = params[0], params[1], params[2]
        return Asym * np.exp(-b2 * b3**x)

    def gradient(
        self, params: NDArray[np.floating], x: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        Asym, b2, b3 = params[0], params[1], params[2]
        b3_x = b3**x
        exp_term = np.exp(-b2 * b3_x)

        grad = np.zeros((len(x), 3), dtype=np.float64)
        grad[:, 0] = exp_term
        grad[:, 1] = -Asym * b3_x * exp_term
        grad[:, 2] = -Asym * b2 * x * b3 ** (x - 1) * exp_term

        return grad

    def get_start(self, x: NDArray[np.floating], y: NDArray[np.floating]) -> NDArray[np.floating]:
        Asym = np.max(y)
        b2 = 1.0
        b3 = 0.5
        return np.array([Asym, b2, b3], dtype=np.float64)


@dataclass
class SSbiexp(NonlinearModel):
    @property
    def name(self) -> str:
        return "SSbiexp"

    @property
    def param_names(self) -> list[str]:
        return ["A1", "lrc1", "A2", "lrc2"]

    def predict(
        self, params: NDArray[np.floating], x: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        A1, lrc1, A2, lrc2 = params[0], params[1], params[2], params[3]
        return A1 * np.exp(-np.exp(lrc1) * x) + A2 * np.exp(-np.exp(lrc2) * x)

    def gradient(
        self, params: NDArray[np.floating], x: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        A1, lrc1, A2, lrc2 = params[0], params[1], params[2], params[3]
        rc1, rc2 = np.exp(lrc1), np.exp(lrc2)
        exp1, exp2 = np.exp(-rc1 * x), np.exp(-rc2 * x)

        grad = np.zeros((len(x), 4), dtype=np.float64)
        grad[:, 0] = exp1
        grad[:, 1] = -A1 * rc1 * x * exp1
        grad[:, 2] = exp2
        grad[:, 3] = -A2 * rc2 * x * exp2

        return grad

    def get_start(self, x: NDArray[np.floating], y: NDArray[np.floating]) -> NDArray[np.floating]:
        A1 = np.max(y) / 2
        A2 = np.max(y) / 2
        lrc1 = np.log(0.1)
        lrc2 = np.log(0.01)
        return np.array([A1, lrc1, A2, lrc2], dtype=np.float64)


class CustomModel(NonlinearModel):
    def __init__(
        self,
        predict_fn: Callable[[NDArray[np.floating], NDArray[np.floating]], NDArray[np.floating]],
        gradient_fn: Callable[[NDArray[np.floating], NDArray[np.floating]], NDArray[np.floating]]
        | None,
        param_names: list[str],
        name: str = "custom",
        start_fn: Callable[[NDArray[np.floating], NDArray[np.floating]], NDArray[np.floating]]
        | None = None,
    ) -> None:
        self._predict_fn = predict_fn
        self._gradient_fn = gradient_fn
        self._param_names = param_names
        self._name = name
        self._start_fn = start_fn

    @property
    def name(self) -> str:
        return self._name

    @property
    def param_names(self) -> list[str]:
        return self._param_names

    def predict(
        self, params: NDArray[np.floating], x: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        return self._predict_fn(params, x)

    def gradient(
        self, params: NDArray[np.floating], x: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        if self._gradient_fn is not None:
            return self._gradient_fn(params, x)

        eps = 1e-7
        n = len(x)
        p = len(params)
        grad = np.zeros((n, p), dtype=np.float64)

        for j in range(p):
            params_plus = params.copy()
            params_plus[j] += eps
            params_minus = params.copy()
            params_minus[j] -= eps
            grad[:, j] = (self._predict_fn(params_plus, x) - self._predict_fn(params_minus, x)) / (
                2 * eps
            )

        return grad

    def get_start(self, x: NDArray[np.floating], y: NDArray[np.floating]) -> NDArray[np.floating]:
        if self._start_fn is not None:
            return self._start_fn(x, y)
        return np.ones(self.n_params, dtype=np.float64)
