"""Frontend for the publicly released ASID exploration objective."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .backends.asid_numpy import ASIDBackendNumpy


class ASIDSolver:
    """Evaluate informative actions against a task-provided parameterized simulator.

    The task owns the simulator, public observations, and candidate actions. The
    backend implements the finite-difference objective in the authors' release.
    """

    def __init__(
        self,
        transition: Callable[[Any, np.ndarray, np.ndarray], np.ndarray],
        *,
        delta: float = 0.1,
        normalization: float = 1.0,
        max_reward: float = 1000.0,
    ) -> None:
        self._backend_impl = ASIDBackendNumpy(
            transition, delta=delta, normalization=normalization,
            max_reward=max_reward,
        )

    @property
    def backend(self) -> ASIDBackendNumpy:
        return self._backend_impl

    def information_reward(
        self, state: Any, action: np.ndarray, parameters: np.ndarray
    ) -> float:
        return self._backend_impl.information_reward(state, action, parameters)

    def observation_jacobian(
        self, state: Any, action: np.ndarray, parameters: np.ndarray
    ) -> np.ndarray:
        return self._backend_impl.observation_jacobian(state, action, parameters)


__all__ = ["ASIDSolver"]
