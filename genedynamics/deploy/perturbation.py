"""
Perturbation (push recovery) support for deploy pipeline.

Wraps env to apply external impulses at configurable intervals for push_recovery task.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from genedynamics.deploy.task_config import PerturbationConfig


class PushRecoveryEnvWrapper:
    """
    Env wrapper that applies external impulses for push recovery evaluation.

    Injects impulse into base velocity (qvel[:3] or body-specific) at
    configurable step intervals. Used when task.perturbation is set.
    """

    def __init__(
        self,
        env: Any,
        perturbation: PerturbationConfig,
        rng: Optional[Any] = None,
    ):
        self._env = env
        self._pert = perturbation
        self._rng = rng
        self._step_count = 0
        self._next_impulse_step: Optional[int] = None
        self._nq = getattr(env, "nq", 15)
        self._nv = getattr(env, "nv", 14)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def _sample_impulse_interval(self) -> int:
        """Sample next impulse step offset."""
        interval = self._pert.interval_steps
        if interval is None or len(interval) < 2:
            return 30 + int(np.random.randint(0, 50))
        lo, hi = int(interval[0]), int(interval[1])
        return lo + int(np.random.randint(0, max(1, hi - lo)))

    def _sample_direction(self) -> np.ndarray:
        """Sample impulse direction (unit vector)."""
        if self._pert.fixed_direction is not None:
            d = np.asarray(self._pert.fixed_direction, dtype=np.float32)
            n = np.linalg.norm(d)
            return (d / n) if n > 1e-8 else np.array([1, 0, 0], dtype=np.float32)
        if self._pert.direction_randomize:
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            return np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi),
            ], dtype=np.float32)
        return np.array([1, 0, 0], dtype=np.float32)

    def _apply_impulse(self, state: np.ndarray) -> np.ndarray:
        """Add impulse to state velocity (in-place on copy)."""
        state = np.asarray(state, dtype=np.float32).ravel().copy()
        mag = float(self._pert.impulse_magnitude)
        direction = self._sample_direction()
        impulse = mag * direction
        if state.size >= self._nq + 3:
            state[self._nq : self._nq + 3] += impulse
        return state

    def reset(self, rng: Optional[Any] = None, seed: Optional[int] = None, **kwargs: Any) -> Tuple[np.ndarray, Dict]:
        """Reset inner env and schedule next impulse."""
        out = self._env.reset(rng=rng, seed=seed, **kwargs)
        self._step_count = 0
        self._next_impulse_step = self._sample_impulse_interval()
        return out

    def transition(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Step with optional impulse injection before transition."""
        state = np.asarray(state, dtype=np.float32).ravel()
        if self._next_impulse_step is not None and self._step_count >= self._next_impulse_step:
            state = self._apply_impulse(state)
            self._next_impulse_step = self._step_count + self._sample_impulse_interval()
        self._step_count += 1
        return self._env.transition(state, action)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step interface (Gymnasium-style) if env supports it."""
        if hasattr(self._env, "step"):
            return self._env.step(action)
        state = getattr(self._env, "_last_state", None)
        if state is None:
            raise RuntimeError("PushRecoveryEnvWrapper.step requires env to expose _last_state or step")
        next_state = self.transition(state, action)
        cost = self._env.cost(next_state) if hasattr(self._env, "cost") else 0.0
        return next_state, cost, False, False, {}


def wrap_env_with_perturbation(
    env: Any,
    perturbation: Optional[PerturbationConfig],
) -> Any:
    """Wrap env with PushRecoveryEnvWrapper when perturbation config is present."""
    if perturbation is None:
        return env
    return PushRecoveryEnvWrapper(env, perturbation)
