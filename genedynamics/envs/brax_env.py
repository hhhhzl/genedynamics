"""
Brax physics environment with flat-state interface for MBD/planning.

Wraps Brax PipelineEnv to expose jax_transition(flat_state, action) -> flat_state,
matching the interface expected by EnvDynamicsAdapter and diffusion solvers.
State format: flat [qpos; qvel], same as Brax obs.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from brax.envs.base import PipelineEnv, State
    BRAX_AVAILABLE = True
except ImportError:
    BRAX_AVAILABLE = False
    jax = None
    jnp = None
    PipelineEnv = None
    State = None


class BraxFlatEnv:
    """
    Wrapper that exposes flat-state jax_transition for Brax PipelineEnv.

    Converts flat [qpos; qvel] <-> Brax State for each step, so MBD/EB-MBD/CFS
    planners can use env.jax_transition(state, action) with flat arrays.
    """

    def __init__(
        self,
        brax_env: Any,
        *,
        target: Tuple[float, float, float] = (2.0, 0.0, 0.5),
        horizon: int = 200,
    ):
        if not BRAX_AVAILABLE:
            raise ImportError(
                "BraxFlatEnv requires brax and jax. pip install brax jax jaxlib"
            )
        self._env = brax_env
        self.target = target
        self.horizon = horizon
        self.dt = float(getattr(brax_env.sys, "dt", 0.02))
        nq = brax_env.sys.q_size()
        nqd = brax_env.sys.qd_size()
        self._nq = nq
        self._nqd = nqd
        self.nq = nq
        self.nv = nqd
        self.act_dim = brax_env.sys.act_size()
        self.state_dim = nq + nqd
        self.control_limit = 1.0
        self.model = "go2" if "go2" in type(brax_env).__name__.lower() else "humanoid"

    def jax_transition(self, state: Any, action: Any) -> Any:
        """Flat state -> step -> flat next state. JAX-compatible."""
        # Sanitize: NaN action -> pipeline explosion -> NaN cascade in diffusion; replace with 0
        action = jnp.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        qpos = state[: self._nq]
        qvel = state[self._nq :]
        pipeline_state = self._env.pipeline_init(qpos, qvel)
        brax_state = State(pipeline_state, state, 0.0, False, {})
        next_brax_state = self._env.step(brax_state, action)
        out = next_brax_state.obs
        # Replace NaN with previous state to prevent propagation (Brax can explode on invalid poses)
        return jnp.where(jnp.any(jnp.isnan(out)), state, out)

    def jax_model_transition(self, state: Any, action: Any) -> Any:
        """Same as jax_transition (no projection)."""
        return self.jax_transition(state, action)

    def transition(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """NumPy transition for rollout/deploy."""
        return np.asarray(
            self.jax_transition(
                jnp.asarray(state, dtype=jnp.float32),
                jnp.asarray(action, dtype=jnp.float32),
            ),
            dtype=np.float32,
        )

    def reset(self, rng: Any = None, seed: Optional[int] = None, **kwargs: Any) -> Tuple[np.ndarray, dict]:
        """Reset to initial state. Returns (flat_state, info)."""
        if rng is None and seed is not None:
            rng = jax.random.PRNGKey(seed)
        if rng is None:
            rng = jax.random.PRNGKey(0)
        brax_state = self._env.reset(rng)
        flat = np.asarray(brax_state.obs, dtype=np.float32)
        return flat, {}


def make_brax_go2(**kwargs) -> BraxFlatEnv:
    """Quadruped Go2 with Brax positional physics (self-contained, no third_party)."""
    from genedynamics.envs.quadruped_brax import QuadrupedGo2BraxEnv
    env = QuadrupedGo2BraxEnv()
    brax_kw = {k: v for k, v in kwargs.items() if k in ("target", "horizon")}
    return BraxFlatEnv(env, **brax_kw)


def make_brax_humanoid_run(**kwargs) -> BraxFlatEnv:
    """Humanoid run with Brax positional physics (self-contained, no third_party)."""
    from genedynamics.envs.humanoid_brax import HumanoidRunBraxEnv
    env = HumanoidRunBraxEnv()
    brax_kw = {k: v for k, v in kwargs.items() if k in ("target", "horizon")}
    return BraxFlatEnv(env, **brax_kw)
