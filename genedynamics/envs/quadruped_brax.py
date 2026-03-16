"""
Quadruped Brax environments (Go2) - self-contained, no third_party.

Brax positional physics for MBD planning. State: flat [qpos; qvel].
Action: [-1, 1] mapped to joint position targets via act2joint.
"""

from __future__ import annotations

from functools import partial
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


def _get_go2_model_path() -> Path:
    """Path to bundled Go2 MJCF (position actuators, MJX-compatible collisions)."""
    env_dir = Path(__file__).resolve().parent
    path = env_dir / "assets" / "brax" / "unitree_go2" / "go2_position.xml"
    if not path.exists():
        raise FileNotFoundError(
            f"Go2 Brax model not found at {path}. "
            "Ensure genedynamics/envs/assets/brax/unitree_go2/ exists."
        )
    return path


# Go2 home pose from keyframe (Brax mjcf.load does not load keyframe; joints default to 0).
_GO2_HOME_QPOS = [
    0.0, 0.0, 0.27, 1.0, 0.0, 0.0, 0.0,  # root: xyz, quat wxyz
    0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8,  # 12 joints
]


class QuadrupedGo2BraxEnv(PipelineEnv):
    """Go2 quadruped with Brax positional physics. MBD action [-1,1] -> joint targets."""

    def __init__(self, n_frames: int = 4):
        if not BRAX_AVAILABLE:
            raise ImportError("brax and jax required. pip install brax jax jaxlib")
        model_path = _get_go2_model_path()
        sys = mjcf.load(str(model_path))
        # Brax mjcf.load uses zeros for joints; override with home keyframe.
        try:
            import mujoco
            mj = mujoco.MjModel.from_xml_path(str(model_path))
            kf = mj.keyframe("home")
            init_q = jnp.array(kf.qpos, dtype=jnp.float32)
        except Exception:
            init_q = jnp.array(_GO2_HOME_QPOS, dtype=jnp.float32)
        if init_q.shape[0] == sys.q_size():
            sys = sys.tree_replace({"init_q": init_q})
        self._joint_range = jnp.array(sys.jnt_range[1:])  # skip root freejoint
        super().__init__(sys=sys, backend="positional", n_frames=n_frames)

    @partial(jax.jit, static_argnums=(0,))
    def act2joint(self, act: jax.Array) -> jax.Array:
        """Map MBD action [-1, 1] to joint position targets (radians)."""
        act_norm = (act + 1.0) / 2.0
        targets = (
            self._joint_range[:, 0]
            + act_norm * (self._joint_range[:, 1] - self._joint_range[:, 0])
        )
        return jnp.clip(targets, self._joint_range[:, 0], self._joint_range[:, 1])

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
        ctrl = self.act2joint(action)
        pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)
        obs = self._get_obs(pipeline_state, action)
        reward = self._get_reward(pipeline_state)
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward)

    def _get_obs(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
        return jnp.concatenate([pipeline_state.q, pipeline_state.qd], axis=-1)

    def _get_reward(self, pipeline_state: base.State) -> jax.Array:
        forward_vel = pipeline_state.xd.vel[0, 0]
        height = pipeline_state.x.pos[0, 2]
        height_penalty = -jnp.clip(jnp.abs(height - 0.27), 0.0, 0.3) * 0.3
        lateral_penalty = -jnp.abs(pipeline_state.x.pos[0, 1]) * 0.05
        return forward_vel * 2.0 + height_penalty + lateral_penalty
