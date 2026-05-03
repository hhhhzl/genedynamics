"""Terrain-locomotion task wrapper (writeup §11 Task 1).

Reward (matches the existing crawling_ground objective):
    R = Σ_t (com_x[t] - com_x[0])  +  shaping * (com_x[T] - com_x[0])
        - λ_back * Σ_t max(0, -v_x[t])

Success: Δx_COM > ε AND robot did not flip (placeholder: not yet detecting flip).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import jax.numpy as jnp

from ..scene import MPMConfig, SceneData, rollout_return
from ..terrain import to_grid
from .regime import RegimeSpec


@dataclass(frozen=True)
class LocomotionTask:
    """Static config for the locomotion task."""

    success_threshold_x: float = 0.02   # Δx_COM > 2% domain length to count
    n_grid: int = 64

    @property
    def name(self) -> str:
        return "locomotion"


def evaluate_locomotion(
    task: LocomotionTask,
    regime: RegimeSpec,
    *,
    x_morph: jnp.ndarray,
    phi: jnp.ndarray,
    scene: SceneData,
    cfg: MPMConfig,
    num_env_steps: int,
    E0: float = 1.0,
) -> Dict[str, Any]:
    """Run one locomotion rollout under the given regime; return metrics dict.

    Returns
    -------
    {
        "reward":      float
        "delta_x":     float    # final COM forward displacement
        "success":     bool     # delta_x > task.success_threshold_x
        "com_x_traj":  (T, 3) jnp.ndarray
    }
    """
    if regime.has_manipuland:
        raise ValueError("LocomotionTask received a regime with a manipuland; "
                         "use PushTask for push regimes.")
    terrain_h = jnp.asarray(to_grid(regime.terrain, task.n_grid))
    friction = jnp.asarray(regime.friction, dtype=jnp.float32)
    reward, final_disp, com_traj = rollout_return(
        x_morph, phi, friction, scene, cfg, num_env_steps,
        E0=E0, terrain_height=terrain_h,
    )
    success = final_disp > jnp.asarray(task.success_threshold_x, dtype=jnp.float32)
    return {
        "reward": reward,
        "delta_x": final_disp,
        "success": success,
        "com_x_traj": com_traj,
    }
