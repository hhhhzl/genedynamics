"""Terrain-aware object-pushing task (writeup §11 Task 2).

Reward (writeup §11):
    R = w1 * (d_0 - d_T)              # closing distance object → goal
      + w2 * Δx_obj                   # absolute object forward displacement
      - w3 * Σ‖u_t‖²                  # control penalty (proxy: ‖φ‖² scalar)
      - w5 * 1[Δx_obj < ε]            # stuck robot penalty

Success: |obj − goal| < ε_d at terminal step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import jax.numpy as jnp

from ..scene import MPMConfig, SceneData, rollout_return_push
from ..terrain import to_grid
from ..manipuland import ManipulandConfig
from .regime import RegimeSpec


@dataclass(frozen=True)
class PushTask:
    """Static config for the push task."""

    goal_x: float = 0.85                       # world-x goal position
    success_distance: float = 0.05             # |obj - goal| < this counts as success
    reward_weights: Tuple[float, float, float, float] = (1.0, 5.0, 0.01, 50.0)
    n_grid: int = 64

    @property
    def name(self) -> str:
        return "push"


def evaluate_push(
    task: PushTask,
    regime: RegimeSpec,
    *,
    x_morph: jnp.ndarray,
    phi: jnp.ndarray,
    scene: SceneData,
    cfg: MPMConfig,
    num_env_steps: int,
    E0: float = 1.0,
) -> Dict[str, Any]:
    """Run one push rollout under the given regime; return metrics dict."""
    if not regime.has_manipuland:
        raise ValueError("PushTask requires a regime with a manipuland; "
                         "got a locomotion-only regime.")
    terrain_h = jnp.asarray(to_grid(regime.terrain, task.n_grid))
    friction = jnp.asarray(regime.friction, dtype=jnp.float32)
    reward, dT, com_traj, obj_traj = rollout_return_push(
        x_morph, phi, friction, scene, cfg, num_env_steps,
        manip_cfg=regime.manipuland,
        goal_x=task.goal_x,
        E0=E0,
        terrain_height=terrain_h,
        weights=task.reward_weights,
    )
    success = dT < jnp.asarray(task.success_distance, dtype=jnp.float32)
    return {
        "reward": reward,
        "dist_to_goal": dT,
        "success": success,
        "com_x_traj": com_traj,
        "obj_x_traj": obj_traj,
    }
