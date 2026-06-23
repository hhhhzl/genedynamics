"""Brax PipelineEnv base for legged robots with position-stiffness (PD) control.

Faithful port of dial-mpc's ``BaseEnv`` + ``BaseEnvConfig`` + the foot-step /
body-velocity utilities, restructured to live in genedynamics. This is the brax
physics base the DIAL h1/go2 task envs (``domains/humanoid/h1_brax.py``,
``domains/quadruped/go2_brax.py``) build on. It is a real brax ``PipelineEnv``
(``reset``/``step`` return brax ``State``, ``n_frames`` substeps, mjx backend),
which the genedynamics DIAL backend rolls out via ``env.step`` (reading
``state.reward``) -- the stable physics path validated against dial-mpc.

Assets are vendored under ``genedynamics/envs/assets/<robot>/`` so this is
self-contained (no external robot-model runtime dependency).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Union

import jax
import jax.numpy as jnp

try:
    from brax import math
    from brax.base import System
    from brax.envs.base import PipelineEnv
    BRAX_AVAILABLE = True
except ImportError:  # pragma: no cover - brax optional; envs that use this base
    BRAX_AVAILABLE = False  # are imported under guarded try/except (skipped on CPU)
    math = None
    System = object
    PipelineEnv = object


_ASSET_ROOT = Path(__file__).resolve().parent / "assets"


def get_model_path(robot_name: str, model_name: str) -> Path:
    """Vendored robot MJCF path (organized by robot: ``assets/<robot>/<model>``)."""
    p = _ASSET_ROOT / robot_name / model_name
    if not p.exists():
        raise FileNotFoundError(
            f"asset not found: {p}. Expected under {_ASSET_ROOT}/{robot_name}/."
        )
    return p


# --- utilities (port of dial_mpc.utils.function_utils) ---

def global_to_body_velocity(v, q):
    """Transform a global velocity into the body frame (inverse-rotate by q)."""
    return math.inv_rotate(v, q)


def body_to_global_velocity(v, q):
    return math.rotate(v, q)


def get_foot_step(duty_ratio, cadence, amplitude, phases, time):
    """Target swing-foot height per leg (verbatim dial-mpc get_foot_step)."""
    def step_height(t, footphase, duty_ratio):
        angle = (t + jnp.pi - footphase) % (2 * jnp.pi) - jnp.pi
        angle = jnp.where(duty_ratio < 1, angle * 0.5 / (1 - duty_ratio), angle)
        clipped_angle = jnp.clip(angle, -jnp.pi / 2, jnp.pi / 2)
        value = jnp.where(duty_ratio < 1, jnp.cos(clipped_angle), 0)
        final_value = jnp.where(jnp.abs(value) >= 1e-6, jnp.abs(value), 0.0)
        return final_value

    h_steps = amplitude * jax.vmap(step_height, in_axes=(None, 0, None))(
        time * 2 * jnp.pi * cadence + jnp.pi,
        2 * jnp.pi * phases,
        duty_ratio,
    )
    return h_steps


# --- config + base env (port of dial_mpc base_env_config + base_env) ---

@dataclass
class BaseEnvConfig:
    task_name: str = "default"
    randomize_tasks: bool = False
    kp: Union[float, Any] = 30.0
    kd: Union[float, Any] = 1.0
    debug: bool = False
    dt: float = 0.02       # env (control) step
    timestep: float = 0.02  # underlying simulator step; dt must be divisible by it
    backend: str = "mjx"
    leg_control: str = "torque"  # "torque" (act2tau PD) or "position" (act2joint)
    action_scale: float = 1.0


class BaseEnv(PipelineEnv):
    """Brax PipelineEnv base with act2joint / act2tau position-stiffness control."""

    def __init__(self, config: BaseEnvConfig):
        if not BRAX_AVAILABLE:
            raise ImportError("brax + jax required. pip install brax jax jaxlib")
        assert jnp.allclose(config.dt % config.timestep, 0.0), "dt must be divisible by timestep"
        self._config = config
        n_frames = int(config.dt / config.timestep)
        sys = self.make_system(config)
        super().__init__(sys, config.backend, n_frames, config.debug)

        self.physical_joint_range = self.sys.jnt_range[1:]
        self.joint_range = self.physical_joint_range
        self.joint_torque_range = self.sys.actuator_ctrlrange
        self._nv = self.sys.nv
        self._nq = self.sys.nq

    def make_system(self, config: BaseEnvConfig) -> System:
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def act2joint(self, act: "jax.Array") -> "jax.Array":
        act_normalized = (act * self._config.action_scale + 1.0) / 2.0  # -> [0, 1]
        joint_targets = self.joint_range[:, 0] + act_normalized * (
            self.joint_range[:, 1] - self.joint_range[:, 0]
        )
        joint_targets = jnp.clip(
            joint_targets,
            self.physical_joint_range[:, 0],
            self.physical_joint_range[:, 1],
        )
        return joint_targets

    @partial(jax.jit, static_argnums=(0,))
    def act2tau(self, act: "jax.Array", pipline_state) -> "jax.Array":
        joint_target = self.act2joint(act)
        q = pipline_state.qpos[7:]
        q = q[: len(joint_target)]
        qd = pipline_state.qvel[6:]
        qd = qd[: len(joint_target)]
        q_err = joint_target - q
        tau = self._config.kp * q_err - self._config.kd * qd
        tau = jnp.clip(tau, self.joint_torque_range[:, 0], self.joint_torque_range[:, 1])
        return tau

    # --- MDAC constraint hook (default no-op) ------------------------------
    # Constrained envs override this to expose the manifold residual the MDAC
    # solver reads (soft-feasibility / geometry). It receives the full
    # brax ``State`` (so it can read both ``state.pipeline_state`` for the clean
    # predicted kinematics AND ``state.info`` for env-carried context such as
    # surface coordinates) and is evaluated on that clean state — never
    # differentiated through the mjx rollout. Returns (h, g): equality residual h
    # (== 0 feasible) and inequality residual g (<= 0 feasible). The unconstrained
    # default returns zero-width arrays, so MDAC degenerates to DIAL.
    def constraint_residual(self, state, action, ctx=None):
        z = jnp.zeros((0,), dtype=jnp.float32)
        return z, z
