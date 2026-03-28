"""
Stepping-stones kinematics helpers.

State formats supported:
- 12D hybrid anchor-length planner:
    [b_x, b_y, psi, v_x, v_y, omega, x_L, x_R, l_L, l_R, mode, tau]
- 4D legacy pair planner:
    [pL_x, pL_y, pR_x, pR_y]
- 3D legacy midline planner:
    [m_x, m_y, psi]
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

LEG_NAMES = ("FL", "FR", "RL", "RR")
MODE_DS_FL_RR = 0
MODE_QS_AFTER_FL_RR = 1
MODE_DS_FR_RL = 2
MODE_QS_AFTER_FR_RL = 3


def wrap_angle(theta: np.ndarray | float) -> np.ndarray | float:
    return np.arctan2(np.sin(theta), np.cos(theta))


def normal_from_yaw(yaw: np.ndarray | float) -> np.ndarray:
    y = np.asarray(yaw, dtype=np.float32)
    return np.stack([-np.sin(y), np.cos(y)], axis=-1)


def states_to_lr_pairs(
    states: np.ndarray,
    *,
    step_width: float = 0.30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert trajectory states to left/right pair placements.

    Supports 12D/11D quadruped foothold states and legacy 4D/3D states.
    """
    body, _yaw, feet, _phase = decode_plan_states(states, step_width=step_width)
    p_l = 0.5 * (feet["FL"] + feet["RL"])
    p_r = 0.5 * (feet["FR"] + feet["RR"])
    return p_l.astype(np.float32), p_r.astype(np.float32), body.astype(np.float32)


def pair_to_virtual_feet(
    p_l: np.ndarray,
    p_r: np.ndarray,
    yaw: np.ndarray,
    *,
    half_pair_length: float = 0.18,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Map left/right pair points to four virtual feet (LF, RF, LH, RH).
    """
    p_l = np.asarray(p_l, dtype=np.float32)
    p_r = np.asarray(p_r, dtype=np.float32)
    yaw = np.asarray(yaw, dtype=np.float32).reshape(-1)
    ex = np.stack([np.cos(yaw), np.sin(yaw)], axis=-1)
    d = float(half_pair_length) * ex
    lf = p_l + d
    lh = p_l - d
    rf = p_r + d
    rh = p_r - d
    return lf, rf, lh, rh


def decode_plan_states(
    states: np.ndarray,
    *,
    step_width: float = 0.30,
    half_pair_length: float = 0.18,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    s = np.asarray(states, dtype=np.float32)
    if s.ndim == 1:
        s = s.reshape(1, -1)
    if s.ndim != 2 or s.shape[1] < 3:
        raise ValueError(f"Expected state dim >= 3, got {s.shape}")

    if s.shape[1] == 12:
        body = s[:, :2].astype(np.float32)
        yaw = s[:, 2].astype(np.float32)
        x_l = s[:, 6].astype(np.float32)
        x_r = s[:, 7].astype(np.float32)
        l_l = s[:, 8].astype(np.float32)
        l_r = s[:, 9].astype(np.float32)
        mode = np.mod(np.rint(s[:, 10]).astype(np.int32), 4)
        half_w = 0.5 * float(step_width)

        fl = np.zeros((s.shape[0], 2), dtype=np.float32)
        fr = np.zeros((s.shape[0], 2), dtype=np.float32)
        rl = np.zeros((s.shape[0], 2), dtype=np.float32)
        rr = np.zeros((s.shape[0], 2), dtype=np.float32)

        left_y = np.full((s.shape[0],), half_w, dtype=np.float32)
        right_y = np.full((s.shape[0],), -half_w, dtype=np.float32)

        mask_flrr = np.logical_or(mode == MODE_DS_FL_RR, mode == MODE_QS_AFTER_FL_RR)
        mask_frrl = np.logical_or(mode == MODE_DS_FR_RL, mode == MODE_QS_AFTER_FR_RL)

        fl[mask_flrr, 0] = x_l[mask_flrr]
        fl[mask_flrr, 1] = left_y[mask_flrr]
        rl[mask_flrr, 0] = x_l[mask_flrr] - l_l[mask_flrr]
        rl[mask_flrr, 1] = left_y[mask_flrr]
        rr[mask_flrr, 0] = x_r[mask_flrr]
        rr[mask_flrr, 1] = right_y[mask_flrr]
        fr[mask_flrr, 0] = x_r[mask_flrr] + l_r[mask_flrr]
        fr[mask_flrr, 1] = right_y[mask_flrr]

        rl[mask_frrl, 0] = x_l[mask_frrl]
        rl[mask_frrl, 1] = left_y[mask_frrl]
        fl[mask_frrl, 0] = x_l[mask_frrl] + l_l[mask_frrl]
        fl[mask_frrl, 1] = left_y[mask_frrl]
        fr[mask_frrl, 0] = x_r[mask_frrl]
        fr[mask_frrl, 1] = right_y[mask_frrl]
        rr[mask_frrl, 0] = x_r[mask_frrl] - l_r[mask_frrl]
        rr[mask_frrl, 1] = right_y[mask_frrl]

        feet = {"FL": fl, "FR": fr, "RL": rl, "RR": rr}
        return body, yaw, feet, mode

    if s.shape[1] >= 4:
        p_l = s[:, :2].astype(np.float32)
        p_r = s[:, 2:4].astype(np.float32)
        body = 0.5 * (p_l + p_r)
        yaw = estimate_yaw_from_mid(body)
        fl, fr, rl, rr = pair_to_virtual_feet(
            p_l,
            p_r,
            yaw,
            half_pair_length=half_pair_length,
        )
        feet = {"FL": fl, "FR": fr, "RL": rl, "RR": rr}
        phase = (np.arange(s.shape[0], dtype=np.int32) % 4)
        return body.astype(np.float32), yaw.astype(np.float32), feet, phase

    body = s[:, :2].astype(np.float32)
    yaw = s[:, 2].astype(np.float32)
    n = normal_from_yaw(yaw)
    half_w = 0.5 * float(step_width)
    p_l = body + half_w * n
    p_r = body - half_w * n
    fl, fr, rl, rr = pair_to_virtual_feet(
        p_l,
        p_r,
        yaw,
        half_pair_length=half_pair_length,
    )
    feet = {"FL": fl, "FR": fr, "RL": rl, "RR": rr}
    phase = (np.arange(s.shape[0], dtype=np.int32) % 4)
    return body.astype(np.float32), yaw.astype(np.float32), feet, phase


def estimate_yaw_from_mid(mid: np.ndarray, default: float = 0.0) -> np.ndarray:
    """
    Robust tangent-based heading estimate for a midline trajectory.
    """
    m = np.asarray(mid, dtype=np.float32)
    if m.ndim != 2 or m.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if m.shape[0] == 1:
        return np.asarray([default], dtype=np.float32)
    v = np.zeros_like(m, dtype=np.float32)
    v[:-1] = m[1:] - m[:-1]
    v[-1] = v[-2]
    yaw = np.arctan2(v[:, 1], v[:, 0]).astype(np.float32)
    bad = ~np.isfinite(yaw)
    if np.any(bad):
        yaw[bad] = float(default)
    return yaw


def stepping_scene_to_dict(scene: object) -> Optional[dict]:
    if scene is None:
        return None
    try:
        return {
            "level": int(getattr(scene, "level")),
            "difficulty": str(getattr(scene, "difficulty")),
            "map_x": [float(getattr(scene, "map_x")[0]), float(getattr(scene, "map_x")[1])],
            "map_y": [float(getattr(scene, "map_y")[0]), float(getattr(scene, "map_y")[1])],
            "river_x": [float(getattr(scene, "river_x")[0]), float(getattr(scene, "river_x")[1])],
            "has_river": bool(getattr(scene, "has_river", True)),
            "stones_centers": np.asarray(getattr(scene, "stones_centers"), dtype=np.float32).tolist(),
            "stones_radii": np.asarray(getattr(scene, "stones_radii"), dtype=np.float32).tolist(),
        }
    except Exception:
        return None
