"""
Corridor obstacle avoidance trajectory visualisation.

This plugin is now a thin dispatcher: it converts the env + trajectory
into a SceneIR + RobotPoseIR list and hands them to a renderer backend
selected via ``config["renderer"]`` (default ``"matplotlib_top"``).

Backends live in ``visualizations/backends/``. The default backend is
byte-equivalent to the original top-down draw code; iso/pyrender
backends can be added without changing this file.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ...framework.base import VisualizationPlugin
from .scene_ir import corridor_scene_to_ir, corridor_states_to_poses
from .backends.matplotlib_top import render_corridor_top
from .backends.matplotlib_iso import render_corridor_iso

try:
    from .backends.pyrender_mesh import render_corridor_pyrender
    _HAVE_PYRENDER = True
except Exception:  # pyrender / OpenGL may be unavailable on some systems
    render_corridor_pyrender = None
    _HAVE_PYRENDER = False


def _build_corridor_extras(env: Any, states: np.ndarray) -> Dict[str, Any]:
    from genedynamics.envs.humanoid_corridor_2d import (
        TORSO_A, TORSO_B, TORSO_CROUCH_EXTRA,
        H_NOMINAL, H_MIN, H_MAX,
        ARM_REACH_OPEN, ARM_REACH_TUCKED,
    )
    coll = np.zeros(states.shape[0], dtype=bool)
    for t in range(states.shape[0]):
        coll[t] = bool(env.check_collision(states[t]))
    return {
        "torso_a": TORSO_A,
        "torso_b": TORSO_B,
        "torso_crouch_extra": TORSO_CROUCH_EXTRA,
        "h_nominal": H_NOMINAL,
        "h_min": H_MIN,
        "h_max": H_MAX,
        "arm_reach_open": ARM_REACH_OPEN,
        "arm_reach_tucked": ARM_REACH_TUCKED,
        "collision_mask": coll,
    }


# Keys forwarded from yaml `config` into the renderer's `extras` dict.
# Adding a key here is the only thing required to expose a new backend
# knob through yaml.
_FORWARDED_KEYS = (
    # generic
    "n_ghosts",
    "ghost_alpha_min", "ghost_alpha_max",
    # iso backend
    "camera_yaw_deg", "camera_pitch_deg", "auto_resize_fig",
    # pyrender backend
    "render_width", "render_height",
    "camera_yfov_deg", "camera_eye", "camera_target",
    "room_height", "front_wall_height",
    "tall_obstacle_alpha", "short_obstacle_alpha",
    "ribbon_radius", "ribbon_arrow_len", "ribbon_arrow_radius",
)


def _merge_yaml_extras(extras: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Forward known yaml config keys into the backend extras dict.

    yaml config takes precedence over computed extras (so the user can
    override e.g. n_ghosts even though the plugin computes a default).
    """
    if not config:
        return extras
    for key in _FORWARDED_KEYS:
        if key in config:
            extras[key] = config[key]
    return extras


class CorridorTrajectoryVisualizationPlugin(VisualizationPlugin):
    @property
    def name(self) -> str:
        return "corridor_trajectory"

    def visualize(self, fig: Any, ax: Any, data: Dict[str, Any],
                  config: Dict[str, Any]) -> None:
        env = data.get("env")
        traj = data.get("trajectory")
        if env is None or traj is None:
            ax.text(0.5, 0.5, "missing env/trajectory",
                    ha="center", va="center", transform=ax.transAxes)
            return

        if not hasattr(env, "scene"):
            ax.text(0.5, 0.5, "env has no corridor scene",
                    ha="center", va="center", transform=ax.transAxes)
            return

        states = np.asarray(
            [np.asarray(s, dtype=np.float32).ravel() for s in traj.states],
            dtype=np.float32,
        )
        if states.ndim != 2 or states.shape[1] < 14:
            ax.text(0.5, 0.5, "invalid state dim",
                    ha="center", va="center", transform=ax.transAxes)
            return

        scene_ir = corridor_scene_to_ir(env.scene)
        poses = corridor_states_to_poses(states)
        extras = _build_corridor_extras(env, states)
        extras = _merge_yaml_extras(extras, config or {})

        renderer = (config or {}).get("renderer", "matplotlib_top")
        if renderer == "matplotlib_top":
            render_corridor_top(fig, ax, scene_ir, poses, extras=extras)
        elif renderer == "matplotlib_iso":
            render_corridor_iso(fig, ax, scene_ir, poses, extras=extras)
        elif renderer == "pyrender_mesh":
            if not _HAVE_PYRENDER:
                raise RuntimeError(
                    "renderer='pyrender_mesh' requested but pyrender import failed"
                )
            render_corridor_pyrender(fig, ax, scene_ir, poses, extras=extras)
        else:
            raise ValueError(
                f"corridor_trajectory: unknown renderer '{renderer}' "
                f"(available: matplotlib_top, matplotlib_iso, pyrender_mesh)"
            )

    def save(self, output_path: Any, fig: Any, **kwargs: Any) -> None:
        dpi = kwargs.get("dpi", 150)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
