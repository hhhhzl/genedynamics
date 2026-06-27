"""
Humanoid Brax environment (run) - self-contained, no third_party.

Brax positional physics. State: flat [qpos; qvel].
"""

from __future__ import annotations

from pathlib import Path

try:
    import jax
    import jax.numpy as jnp
    from brax import base
    from brax.envs.base import PipelineEnv, State
    from brax.io import mjcf
    BRAX_AVAILABLE = True
except ImportError:
    BRAX_AVAILABLE = False
    jax = None
    jnp = None
    base = None
    PipelineEnv = None
    State = None
    mjcf = None


def _get_humanoid_model_path() -> Path:
    """Path to bundled humanoidrun MJCF."""
    # parents[2] == genedynamics/envs (this file is envs/domains/humanoid/brax.py)
    env_dir = Path(__file__).resolve().parents[2]
    path = env_dir / "assets" / "humanoid_run" / "humanoidrun.xml"
    if not path.exists():
        raise FileNotFoundError(
            f"humanoid_run model not found at {path}. "
            "Ensure genedynamics/envs/assets/humanoid_run/humanoidrun.xml exists."
        )
    return path


class HumanoidRunBraxEnv(PipelineEnv):
    """Humanoid run with Brax positional physics."""

    def __init__(self, n_frames: int = 7):
        if not BRAX_AVAILABLE:
            raise ImportError("brax and jax required. pip install brax jax jaxlib")
        model_path = _get_humanoid_model_path()
        sys = mjcf.load(str(model_path))
        super().__init__(sys=sys, backend="positional", n_frames=n_frames)

    def reset(self, rng: jax.Array) -> State:
        rng, rng1, rng2 = jax.random.split(rng, 3)
        low, hi = -0.01, 0.01
        qpos = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(rng2, (self.sys.qd_size(),), minval=low, maxval=hi)
        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state, jnp.zeros(self.sys.act_size()))
        return State(pipeline_state, obs, 0.0, False, {})

    def step(self, state: State, action: jax.Array) -> State:
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        obs = self._get_obs(pipeline_state, action)
        reward = self._get_reward(pipeline_state)
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward)

    def _get_obs(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
        return jnp.concatenate([pipeline_state.q, pipeline_state.qd], axis=-1)

    def _get_reward(self, pipeline_state: base.State) -> jax.Array:
        return (
            pipeline_state.x.pos[0, 0] * 1.0
            - jnp.clip(jnp.abs(pipeline_state.x.pos[0, 2] - 1.3), -1.0, 1.0) * 1.0
            - jnp.abs(pipeline_state.x.pos[0, 1]) * 0.1
        )
