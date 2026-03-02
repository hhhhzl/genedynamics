from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from genedynamics.envs.external.d3il.specs.base import D3ILTaskSpec, Context


class D3ILTaskEnv:
    """
    Generic wrapper that adapts a D3IL task using a D3ILTaskSpec.

    This is the core “task spec” abstraction (E3): for new tasks, implement a
    new spec instead of duplicating wrapper code.
    """

    def __init__(self, spec: D3ILTaskSpec):
        self.spec = spec
        self._env: Any = None
        self._started: bool = False
        self._ctx: Optional[Context] = None

        # Expose common attributes expected by genedynamics components
        self.dt = float(spec.dt)
        self.horizon = int(spec.horizon)
        self.state_dim = int(spec.state_dim)
        self.act_dim = int(spec.act_dim)
        self.control_limit = float(getattr(spec, "control_limit", 0.0))
        self.target = np.asarray(getattr(spec, "target", np.zeros(2, dtype=np.float32)), dtype=np.float32)

    def _lazy_init(self) -> None:
        if self._env is None:
            self._env = self.spec.make_env()

    def start(self) -> None:
        self._lazy_init()
        if not self._started:
            self.spec.start_env(self._env)
            self._started = True

    def close(self) -> None:
        # Best-effort cleanup; D3IL does not consistently implement close().
        try:
            if self._env is not None and hasattr(self._env, "scene") and hasattr(self._env.scene, "viewer"):
                viewer = getattr(self._env.scene, "viewer", None)
                if viewer is not None and hasattr(viewer, "close"):
                    viewer.close()
        except Exception:
            pass
        self._env = None
        self._started = False
        self._ctx = None

    def reset(self, rng: Optional[Any] = None, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.start()
        state, ctx, info = self.spec.reset_to_state(self._env, rng=rng, **kwargs)
        self._ctx = ctx
        return np.asarray(state, dtype=np.float32), dict(info)

    def step(
        self,
        state: Optional[np.ndarray],
        action: np.ndarray,
        t: Optional[int] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        _ = (state, t)
        if info is None:
            info = {}
        if self._env is None or not self._started or self._ctx is None:
            raise RuntimeError("Call reset() before step()")

        env_action, ctx = self.spec.action_to_env_action(self._env, action, self._ctx)
        next_state, cost, done, ctx, step_info = self.spec.step_to_state(self._env, env_action, ctx)
        self._ctx = ctx

        merged_info = {**info, **step_info}
        return np.asarray(next_state, dtype=np.float32), cost, bool(done), merged_info

    # Optional planner-style helpers
    def transition(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return self.spec.approx_transition(state, action)

    def cost(self, state: np.ndarray) -> float:
        _ = state
        return 0.0


