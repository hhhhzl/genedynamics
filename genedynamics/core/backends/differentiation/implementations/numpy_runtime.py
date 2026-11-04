"""NumPy reference runtime; finite differences are explicitly diagnostic."""

import numpy as np


class NumpyRuntime:
    name = "numpy"
    autodiff = False

    def __init__(self, device="cpu", dtype="float32"):
        if device != "cpu":
            raise ValueError("NumPy reference runtime only supports cpu")
        self.device, self.dtype, self.xp = device, np.dtype(dtype), np

    def array(self, value):
        return np.asarray(value, dtype=self.dtype)

    def to_numpy(self, value):
        return np.asarray(value)

    def scalar(self, value):
        return float(value)

    def copy(self, value):
        return value.copy()

    def clip(self, value, lower, upper):
        return np.clip(value, lower, upper)

    def eye(self, size, like):
        return np.eye(size, dtype=like.dtype)

    def normal(self, shape, seed):
        return np.random.default_rng(seed).standard_normal(shape).astype(self.dtype)

    def value_and_grad(self, function, value):
        # Deliberately not advertised as an automatic-differentiation backend.
        step = 1e-4 if value.dtype == np.float64 else 1e-3
        gradient = np.zeros_like(value)
        for index in np.ndindex(value.shape):
            perturbation = np.zeros_like(value)
            perturbation[index] = step
            gradient[index] = (function(value + perturbation) - function(value - perturbation)) / (2 * step)
        return function(value), gradient

    def jacobian(self, function, value):
        output = np.asarray(function(value))
        rows = [self.value_and_grad(lambda control, row=row: function(control).reshape(-1)[row], value)[1]
                for row in range(output.size)]
        return np.stack(rows).reshape(output.shape + value.shape)
