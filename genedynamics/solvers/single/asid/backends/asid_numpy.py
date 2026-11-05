"""NumPy implementation of the public ASID Fisher-information reward.

The authors' release computes a central-difference observation Jacobian for
each physical parameter, divides it by a scalar normalization, and clips the
squared Frobenius norm of that Jacobian. Simulator state restoration is owned by
the task-provided pure transition callback.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


class ASIDBackendNumpy:
    def __init__(
        self,
        transition: Callable[[Any, np.ndarray, np.ndarray], np.ndarray],
        *,
        delta: float = 0.1,
        normalization: float = 1.0,
        max_reward: float = 1000.0,
    ) -> None:
        if delta <= 0 or normalization <= 0 or max_reward <= 0:
            raise ValueError("delta, normalization, and max_reward must be positive")
        self.transition = transition
        self.delta = float(delta)
        self.normalization = float(normalization)
        self.max_reward = float(max_reward)

    def observation_jacobian(
        self, state: Any, action: np.ndarray, parameters: np.ndarray
    ) -> np.ndarray:
        theta = np.asarray(parameters, dtype=np.float64)
        if theta.ndim != 1 or theta.size == 0 or not np.all(np.isfinite(theta)):
            raise ValueError("parameters must be a finite nonempty vector")
        action = np.asarray(action, dtype=np.float64)
        columns = []
        observation_shape = None
        for index in range(theta.size):
            offset = np.zeros_like(theta)
            offset[index] = self.delta
            plus = np.asarray(self.transition(state, action, theta + offset), dtype=np.float64)
            minus = np.asarray(self.transition(state, action, theta - offset), dtype=np.float64)
            if plus.shape != minus.shape or plus.ndim != 1 or not np.all(np.isfinite(plus)) or not np.all(np.isfinite(minus)):
                raise ValueError("transition must return matching finite observation vectors")
            if observation_shape is None:
                observation_shape = plus.shape
            elif plus.shape != observation_shape:
                raise ValueError("transition observation shape changed across parameters")
            columns.append((plus - minus) / (2.0 * self.delta * self.normalization))
        return np.column_stack(columns)

    def information_reward(
        self, state: Any, action: np.ndarray, parameters: np.ndarray
    ) -> float:
        jacobian = self.observation_jacobian(state, action, parameters)
        return float(np.clip(np.sum(jacobian * jacobian), 0.0, self.max_reward))
