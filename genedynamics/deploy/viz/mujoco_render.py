"""
MuJoCo trajectory rendering for deploy visualization.

Generates GIF from episode states. Used by WebVizService.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def render_episode_to_gif(
    episode_dir: Path,
    states: np.ndarray,
    actions: Optional[np.ndarray] = None,
    *,
    output_path: Optional[Path] = None,
    model: str = "go2",
    width: int = 640,
    height: int = 480,
    fps: float = 20.0,
    draw_trajectory: bool = True,
) -> Path:
    """
    Render episode states as MuJoCo GIF.

    Returns path to generated GIF.
    """
    import mujoco

    states = np.asarray(states, dtype=np.float64)
    if states.ndim == 1:
        states = states.reshape(1, -1)
    last_valid = None
    for i in range(len(states)):
        if np.any(~np.isfinite(states[i])):
            if last_valid is not None:
                states[i] = last_valid.copy()
        else:
            last_valid = states[i].copy()
    if last_valid is None:
        raise ValueError("All states have NaN")

    state_dim = states.shape[1]
    model = "go2" if state_dim >= 35 else "ant"
    positions = states[:, :3]

    model_dir = _get_model_dir(model)
    if model_dir and model_dir.exists():
        tmp_xml = str(model_dir / "_render_traj_temp.xml")
    else:
        fd, tmp_xml = tempfile.mkstemp(suffix=".xml")
        os.close(fd)

    try:
        if model == "go2":
            from genedynamics.envs.utils.mujoco_model_generator import create_go2_render_xml_with_trajectory
            create_go2_render_xml_with_trajectory(
                tmp_xml,
                trajectory_positions=positions if draw_trajectory else [],
            )
        else:
            from genedynamics.envs.utils.mujoco_model_generator import create_ant_render_xml_with_trajectory
            create_ant_render_xml_with_trajectory(
                tmp_xml,
                trajectory_positions=positions if draw_trajectory else [],
            )

        mj_model = mujoco.MjModel.from_xml_path(tmp_xml)
        mj_data = mujoco.MjData(mj_model)
        renderer = mujoco.Renderer(mj_model, height=height, width=width)

        if output_path is None:
            output_path = Path(episode_dir) / "trajectory_mujoco.gif"
        output_path = Path(output_path)

        frames = []
        for i in range(len(states)):
            mj_data.qpos[:] = states[i, : mj_model.nq]
            mj_data.qvel[:] = states[i, mj_model.nq : mj_model.nq + mj_model.nv] if states.shape[1] > mj_model.nq else 0
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(mj_data)
            pixels = renderer.render()
            frames.append(pixels)

        import imageio
        imageio.mimsave(
            str(output_path),
            frames,
            fps=fps,
            loop=0,
        )
        return output_path
    finally:
        try:
            os.unlink(tmp_xml)
        except Exception:
            pass


def _get_model_dir(model: str) -> Optional[Path]:
    if model == "go2":
        try:
            from genedynamics.robots.registry import _get_go2_path
            p = _get_go2_path()
            return Path(p).parent if p else None
        except Exception:
            pass
    else:
        try:
            from genedynamics.robots.registry import _get_ant_path
            p = _get_ant_path()
            return Path(p).parent if p else None
        except Exception:
            pass
    return None
