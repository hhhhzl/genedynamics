"""
Stepping-stones trajectory visualizations.

Like ``corridor_trajectory``, this is now a thin dispatcher: it builds a
SceneIR + RobotPoseIR list and forwards them to a renderer backend
selected via ``config["renderer"]`` (default ``"matplotlib_top"``).

The modes plugin (multi-candidate overlay) draws the background through
the same backend helper, then overlays candidates inline.
"""

from __future__ import annotations

from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

from genedynamics.envs.obstacles.stepping_stones import foot_stepping_violation_np
from genedynamics.tasks.stepping_stones import decode_plan_states

from ...framework.base import VisualizationPlugin
from .scene_ir import (
    SceneIR,
    corridor_scene_to_ir,
    stepping_decoded_to_poses,
    stepping_scene_to_ir,
)
from .backends.matplotlib_top import (
    draw_corridor_scene,
    draw_stepping_scene,
    render_stepping_top,
)
from .backends.matplotlib_iso import render_stepping_iso

try:
    from .backends.pyrender_mesh import render_stepping_pyrender
    _HAVE_PYRENDER = True
except Exception:
    render_stepping_pyrender = None
    _HAVE_PYRENDER = False


def _get_scene(env: Any, obstacles: Any):
    if env is not None and hasattr(env, "scene") and getattr(env, "scene") is not None:
        return getattr(env, "scene")
    if obstacles is not None and hasattr(obstacles, "stepping_scene"):
        return getattr(obstacles, "stepping_scene")
    return None


def _is_corridor_scene(scene: Any) -> bool:
    return hasattr(scene, "corridor_width") and hasattr(scene, "obstacles")


def _scene_to_ir(scene: Any) -> SceneIR:
    if _is_corridor_scene(scene):
        return corridor_scene_to_ir(scene)
    return stepping_scene_to_ir(scene)


def _draw_scene_ir(ax: Any, scene_ir: SceneIR) -> None:
    if scene_ir.kind == "corridor":
        draw_corridor_scene(ax, scene_ir)
    else:
        draw_stepping_scene(ax, scene_ir)


def _draw_scene(ax: Any, scene: Any) -> None:
    """Backward-compat shim for callers (e.g. diffusion plugin) that import
    `_draw_scene` directly. Routes through SceneIR + matplotlib_top backend."""
    _draw_scene_ir(ax, _scene_to_ir(scene))


# Yaml keys forwarded to renderer extras (mirrors corridor plugin).
_FORWARDED_KEYS = (
    "n_ghosts",
    "camera_yaw_deg", "camera_pitch_deg", "auto_resize_fig",
    "render_width", "render_height",
    "camera_yfov_deg", "camera_eye", "camera_target",
    "ribbon_radius", "ribbon_arrow_len", "ribbon_arrow_radius",
)


def _merge_yaml_extras(extras: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    if not config:
        return extras
    for key in _FORWARDED_KEYS:
        if key in config:
            extras[key] = config[key]
    return extras


def _infer_foot_radius(viz_cfg: Dict[str, Any]) -> float:
    if "foot_radius" in viz_cfg:
        return float(viz_cfg["foot_radius"])
    cfg = viz_cfg.get("config", None)
    if cfg is not None and hasattr(cfg, "obstacle_config"):
        obs_cfg = getattr(cfg, "obstacle_config", {}) or {}
        if "robot_radius" in obs_cfg:
            return float(obs_cfg["robot_radius"])
    return 0.03


class SteppingStonesTrajectoryVisualizationPlugin(VisualizationPlugin):
    @property
    def name(self) -> str:
        return "stepping_trajectory"

    def visualize(self, fig: Any, ax: Any, data: Dict[str, Any], config: Dict[str, Any]) -> None:
        traj = data["trajectory"]
        env = data["env"]
        obstacles = data["obstacles"]
        viz_cfg = config or {}
        scene = _get_scene(env, obstacles)
        if scene is None:
            ax.text(0.5, 0.5, "missing stepping scene",
                    ha="center", va="center", transform=ax.transAxes)
            return

        scene_ir = _scene_to_ir(scene)

        states = np.asarray(
            [np.asarray(s, dtype=np.float32).reshape(-1) for s in traj.states],
            dtype=np.float32,
        )
        if not (states.ndim == 2 and states.shape[1] >= 3):
            _draw_scene_ir(ax, scene_ir)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(True, alpha=0.25)
            return

        body, _yaw, feet, mode = decode_plan_states(
            states,
            step_width=float(getattr(env, "step_width",
                                     getattr(env, "stance_width", 0.30))),
            half_pair_length=float(getattr(env, "fore_hind_offset", 0.18)),
            centerline_y=float(getattr(env, "centerline_y", 0.0)),
            env=env,
        )
        poses = stepping_decoded_to_poses(body, feet, mode)

        # Use raw scene arrays (not the IR) for violation checks so behavior
        # matches the original byte-for-byte.
        centers_v = np.asarray(scene.stones_centers, dtype=np.float32)
        radii_v = np.asarray(scene.stones_radii, dtype=np.float32)
        plat_v = np.asarray(
            getattr(scene, "support_platforms", np.zeros((0, 4))),
            dtype=np.float32,
        )

        extras = {
            "foot_radius": _infer_foot_radius(viz_cfg),
            "stone_margin": float(getattr(env, "stone_margin", 0.032)),
            "violation_fn": foot_stepping_violation_np,
            "centers": centers_v,
            "radii": radii_v,
            "platforms": plat_v,
            "partial_until_step": data.get("partial_until_step", None),
            "gif_style": data.get("gif_style", False),
        }
        extras = _merge_yaml_extras(extras, viz_cfg)

        renderer = viz_cfg.get("renderer", "matplotlib_top")
        if renderer == "matplotlib_top":
            render_stepping_top(fig, ax, scene_ir, poses, extras=extras)
        elif renderer == "matplotlib_iso":
            render_stepping_iso(fig, ax, scene_ir, poses, extras=extras)
        elif renderer == "pyrender_mesh":
            if not _HAVE_PYRENDER:
                raise RuntimeError(
                    "renderer='pyrender_mesh' requested but pyrender import failed"
                )
            render_stepping_pyrender(fig, ax, scene_ir, poses, extras=extras)
        else:
            raise ValueError(
                f"stepping_trajectory: unknown renderer '{renderer}' "
                f"(available: matplotlib_top, matplotlib_iso, pyrender_mesh)"
            )

    def save(self, output_path: Any, fig: Any, **kwargs: Any) -> None:
        fig.savefig(output_path, dpi=kwargs.get("dpi", 150),
                    bbox_inches=kwargs.get("bbox_inches", "tight"))


class SteppingStonesModesVisualizationPlugin(VisualizationPlugin):
    @property
    def name(self) -> str:
        return "stepping_modes"

    def visualize(self, fig: Any, ax: Any, data: Dict[str, Any], config: Dict[str, Any]) -> None:
        result = data["result"]
        env = data["env"]
        obstacles = data["obstacles"]
        scene = _get_scene(env, obstacles)
        if scene is None:
            ax.text(0.5, 0.5, "missing stepping scene",
                    ha="center", va="center", transform=ax.transAxes)
            return
        scene_ir = _scene_to_ir(scene)
        _draw_scene_ir(ax, scene_ir)

        partial_t = data.get("partial_until_step", None)

        cand_states = result.get("candidate_states", [])
        if cand_states is None or len(cand_states) == 0:
            cand_states = [result.get("states", [])]
        best_idx = int(result.get("best_idx", 0))
        colors = ["#6a7fdb"] * len(cand_states)
        foot_colors = {"FL": "#ff7f0e", "FR": "#2ca02c",
                       "RL": "#9467bd", "RR": "#8c564b"}
        step_width = float(getattr(env, "step_width",
                                   getattr(env, "stance_width", 0.30)))
        for i, st in enumerate(cand_states):
            s = np.asarray(st, dtype=np.float32)
            if s.ndim != 2 or s.shape[1] < 3:
                continue
            mid, _yaw, feet_i, _phase = decode_plan_states(
                s,
                step_width=step_width,
                half_pair_length=float(getattr(env, "fore_hind_offset", 0.18)),
                centerline_y=float(getattr(env, "centerline_y", 0.0)),
                env=env,
            )
            T_vis = (min(int(partial_t) + 1, mid.shape[0])
                     if partial_t is not None else mid.shape[0])
            is_best = i == best_idx
            lw = 2.6 if is_best else 1.1
            a = 0.95 if is_best else 0.35
            ax.plot(mid[:T_vis, 0], mid[:T_vis, 1], color=colors[i],
                    linewidth=lw, alpha=a, zorder=3)
            for leg in ("FL", "FR", "RL", "RR"):
                fp = feet_i[leg][:T_vis]
                f_lw = 1.0 if is_best else 0.5
                f_a = 0.70 if is_best else 0.18
                ax.plot(fp[:, 0], fp[:, 1], color=foot_colors[leg],
                        linewidth=f_lw, alpha=f_a,
                        zorder=2.7 if is_best else 2.2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.25)

    def save(self, output_path: Any, fig: Any, **kwargs: Any) -> None:
        fig.savefig(output_path, dpi=kwargs.get("dpi", 150),
                    bbox_inches=kwargs.get("bbox_inches", "tight"))
