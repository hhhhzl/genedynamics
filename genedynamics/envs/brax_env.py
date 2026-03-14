"""
Brax physics environment with flat-state interface for MBD/planning.

Wraps Brax PipelineEnv to expose jax_transition(flat_state, action) -> flat_state,
matching the interface expected by EnvDynamicsAdapter and diffusion solvers.
State format: flat [qpos; qvel], same as Brax obs.
"""

from __future__ import annotations

from pathlib import Path
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


def _load_mbd_env(name: str) -> Any:
    """Load MBD env from third_party. Returns Brax PipelineEnv."""
    if not BRAX_AVAILABLE:
        raise ImportError("brax and jax required. pip install brax jax jaxlib")
    root = Path(__file__).resolve().parents[2]
    mbd_root = root / "third_party" / "model-based-diffusion"
    if not mbd_root.exists():
        raise FileNotFoundError(
            f"MBD envs not found at {mbd_root}. "
            "Ensure third_party/model-based-diffusion is available."
        )
    import sys
    if str(mbd_root) not in sys.path:
        sys.path.insert(0, str(mbd_root))
    from mbd.envs import get_env
    return get_env(name)


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
        qpos = state[: self._nq]
        qvel = state[self._nq :]
        pipeline_state = self._env.pipeline_init(qpos, qvel)
        brax_state = State(pipeline_state, state, 0.0, False, {})
        next_brax_state = self._env.step(brax_state, action)
        return next_brax_state.obs

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
    """Quadruped Go2 with Brax positional physics."""
    env = _load_mbd_env("go2run")
    brax_kw = {k: v for k, v in kwargs.items() if k in ("target", "horizon")}
    return BraxFlatEnv(env, **brax_kw)


def make_brax_humanoid_run(**kwargs) -> BraxFlatEnv:
    """Humanoid run with Brax positional physics."""
    env = _load_mbd_env("humanoidrun")
    brax_kw = {k: v for k, v in kwargs.items() if k in ("target", "horizon")}
    return BraxFlatEnv(env, **brax_kw)
