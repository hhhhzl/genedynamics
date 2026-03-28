"""
Path B helper: lift 4D stepping-stones foot trajectories to Go2 MuJoCo qpos/qvel/ctrl.

Lives under experiments.common alongside other experiment-adjacent utilities (e.g. d3il_mpc).
Kinematic mapping: base follows mid-foot path, joints stay at menagerie keyframe.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from genedynamics.tasks.stepping_stones import decode_plan_states

try:
    import mujoco
except ImportError:
    mujoco = None  # type: ignore


def load_best_stepping_traj_from_seed_dir(seed_dir: str | Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load planned trajectory and metadata."""
    seed_path = Path(seed_dir)
    traj_path = seed_path / "trajectory" / "trajectory.json"
    if not traj_path.is_file():
        raise FileNotFoundError(traj_path)
    with open(traj_path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    cand = blob.get("candidate_states") or []
    if not cand:
        raise ValueError(f"No candidate_states in {traj_path}")
    idx = int(blob.get("best_idx", 0))
    idx = max(0, min(idx, len(cand) - 1))
    states = np.asarray(cand[idx], dtype=np.float64)
    if states.ndim != 2 or states.shape[1] < 3:
        raise ValueError(f"Expected (T, 3+) states, got {states.shape}")
    return states, blob


def _go2_xml_path() -> str:
    from genedynamics.robots.registry import _get_go2_path

    p = Path(_get_go2_path() or "")
    if not p.is_file():
        raise RuntimeError(
            "Go2 MJCF not found. Clone mujoco_menagerie to third_party/mujoco_menagerie "
            "or set MUJOCO_MENAGERIE_PATH."
        )
    scene = p.parent / "scene.xml"
    return str(scene if scene.is_file() else p)


def _quat_wxyz_yaw(yaw: float) -> np.ndarray:
    h = 0.5 * float(yaw)
    return np.array([np.cos(h), 0.0, 0.0, np.sin(h)], dtype=np.float64)


def interpolate_stepping_states(
    states: np.ndarray,
    plan_dt: float,
    render_fps: float,
) -> np.ndarray:
    """Linear interpolation in time along the planned waypoints."""
    states = np.asarray(states, dtype=np.float64)
    if states.shape[0] < 2:
        return states.copy()
    t = float(plan_dt)
    if t <= 0:
        t = 1.0
    T = states.shape[0]
    total_time = t * (T - 1)
    n_frames = max(T, int(np.ceil(total_time * float(render_fps))) + 1)
    raw_times = np.arange(T, dtype=np.float64) * t
    eval_times = np.linspace(0.0, total_time, n_frames, dtype=np.float64)
    out = np.zeros((n_frames, 4), dtype=np.float64)
    for i in range(4):
        out[:, i] = np.interp(eval_times, raw_times, states[:, i])
    return out


def go2_kinematic_states_from_stepping(
    stepping_states: np.ndarray,
    *,
    plan_dt: float = 1.0,
    render_fps: float = 50.0,
    step_width: float = 0.30,
    model_xml_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        qpos: (N, nq)
        qvel: (N, nv) — zeros (kinematic replay; sufficient for MuJoCo frame rendering)
        ctrl: (N, nu) — zeros
    """
    if mujoco is None:
        raise ImportError("mujoco is required for Path B Go2 lift")

    states = interpolate_stepping_states(stepping_states, plan_dt=plan_dt, render_fps=render_fps)
    path = model_xml_path or _go2_xml_path()
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    key = 0 if model.nkey > 0 else -1
    if key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key)
    else:
        mujoco.mj_resetData(model, data)
    q_home = np.array(data.qpos, copy=True)
    nq, nv, nu = model.nq, model.nv, model.nu

    mid, yaw, _feet, _phase = decode_plan_states(
        states,
        step_width=step_width,
        half_pair_length=0.18,
    )
    yaw = np.asarray(yaw, dtype=np.float64)

    n = states.shape[0]
    qpos_out = np.zeros((n, nq), dtype=np.float64)
    qvel_out = np.zeros((n, nv), dtype=np.float64)
    ctrl_out = np.zeros((n, nu), dtype=np.float64)

    z_ref = float(q_home[2])
    for i in range(n):
        q = np.array(q_home, copy=True)
        q[0] = float(mid[i, 0])
        q[1] = float(mid[i, 1])
        q[2] = z_ref
        q[3:7] = _quat_wxyz_yaw(float(yaw[i]))
        qpos_out[i] = q

    return qpos_out, qvel_out, ctrl_out


def write_pathb_artifacts(
    seed_dir: str | Path,
    qpos: np.ndarray,
    qvel: np.ndarray,
    ctrl: np.ndarray,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    d = Path(seed_dir)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "qpos.npy", np.asarray(qpos, dtype=np.float32))
    np.save(d / "qvel.npy", np.asarray(qvel, dtype=np.float32))
    np.save(d / "ctrl.npy", np.asarray(ctrl, dtype=np.float32))
    payload = {
        "qpos_shape": list(np.asarray(qpos).shape),
        "qvel_shape": list(np.asarray(qvel).shape),
        "ctrl_shape": list(np.asarray(ctrl).shape),
        "map": "go2_kinematic_midfoot_keyframe",
    }
    if extra:
        payload.update(extra)
    with open(d / "execution_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
