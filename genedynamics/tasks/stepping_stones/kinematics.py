"""
Stepping-stones kinematics helpers.

State formats supported:
- 12D hybrid anchor-length planner:
    [b_x, b_y, psi, v_x, v_y, omega, x_L, x_R, l_L, l_R, mode, tau]
- 16D corridor residual planner (quadruped_stepping_stones_2d):
    [s, ey, epsi, v_s, v_y, omega, res(8), mode, tau].
    Body world position is [s, centerline_y + ey]; nominal template (x_f,x_r,y_L,y_R) comes from
    env fields or ``decode_plan_states(..., x_f_nominal=..., ...)``.
- 20D corridor/template planner (template in state; older packed trajectories):
    [s, ey, epsi, v_s, v_y, omega, res(8), x_f, x_r, y_L, y_R, mode, tau].
    World body is [s, centerline_y + ey]; set decode_plan_states(..., centerline_y=...).
- 24D legacy v2 trajectories (sigma + template + residual; sigma ignored for decode):
    Same foot geometry as 20D; columns 6–9 are unused for visualization.
- 4D legacy pair planner:
    [pL_x, pL_y, pR_x, pR_y]
- 3D legacy midline planner:
    [m_x, m_y, psi]
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

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
    centerline_y: float = 0.0,
    env: Optional[Any] = None,
    x_f_nominal: Optional[float] = None,
    x_r_nominal: Optional[float] = None,
    y_L_nominal: Optional[float] = None,
    y_R_nominal: Optional[float] = None,
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

    if s.shape[1] == 20:
        cy = float(centerline_y)
        body = np.stack([s[:, 0], cy + s[:, 1]], axis=-1).astype(np.float32)
        psi = s[:, 2].astype(np.float32)
        res = s[:, 6:14].reshape(-1, 4, 2).astype(np.float32)
        x_f = s[:, 14].astype(np.float32)
        x_r = s[:, 15].astype(np.float32)
        y_L = s[:, 16].astype(np.float32)
        y_R = s[:, 17].astype(np.float32)
        mode2 = np.mod(np.rint(s[:, 18]).astype(np.int32), 2)
        c = np.cos(psi).astype(np.float32)
        sn = np.sin(psi).astype(np.float32)
        local_fl = np.stack([x_f + res[:, 0, 0], y_L + res[:, 0, 1]], axis=-1)
        local_fr = np.stack([x_f + res[:, 1, 0], y_R + res[:, 1, 1]], axis=-1)
        local_rl = np.stack([x_r + res[:, 2, 0], y_L + res[:, 2, 1]], axis=-1)
        local_rr = np.stack([x_r + res[:, 3, 0], y_R + res[:, 3, 1]], axis=-1)
        loc = np.stack([local_fl, local_fr, local_rl, local_rr], axis=1)
        vx = loc[..., 0]
        vy = loc[..., 1]
        wx = c[:, None] * vx - sn[:, None] * vy
        wy = sn[:, None] * vx + c[:, None] * vy
        world = body[:, None, :] + np.stack([wx, wy], axis=-1)
        feet = {
            "FL": world[:, 0].astype(np.float32),
            "FR": world[:, 1].astype(np.float32),
            "RL": world[:, 2].astype(np.float32),
            "RR": world[:, 3].astype(np.float32),
        }
        mode_out = np.where(mode2 == 0, MODE_DS_FL_RR, MODE_DS_FR_RL).astype(np.int32)
        return body, psi, feet, mode_out

    if s.shape[1] == 24:
        # Legacy packed state: [s,ey,epsi, vel(3), sigma(4) ignored, res(8), x_f,x_r,y_L,y_R, mode, tau]
        cy = float(centerline_y)
        body = np.stack([s[:, 0], cy + s[:, 1]], axis=-1).astype(np.float32)
        psi = s[:, 2].astype(np.float32)
        res = s[:, 10:18].reshape(-1, 4, 2).astype(np.float32)
        x_f = s[:, 18].astype(np.float32)
        x_r = s[:, 19].astype(np.float32)
        y_L = s[:, 20].astype(np.float32)
        y_R = s[:, 21].astype(np.float32)
        mode2 = np.mod(np.rint(s[:, 22]).astype(np.int32), 2)
        c = np.cos(psi).astype(np.float32)
        sn = np.sin(psi).astype(np.float32)
        local_fl = np.stack([x_f + res[:, 0, 0], y_L + res[:, 0, 1]], axis=-1)
        local_fr = np.stack([x_f + res[:, 1, 0], y_R + res[:, 1, 1]], axis=-1)
        local_rl = np.stack([x_r + res[:, 2, 0], y_L + res[:, 2, 1]], axis=-1)
        local_rr = np.stack([x_r + res[:, 3, 0], y_R + res[:, 3, 1]], axis=-1)
        loc = np.stack([local_fl, local_fr, local_rl, local_rr], axis=1)
        vx = loc[..., 0]
        vy = loc[..., 1]
        wx = c[:, None] * vx - sn[:, None] * vy
        wy = sn[:, None] * vx + c[:, None] * vy
        world = body[:, None, :] + np.stack([wx, wy], axis=-1)
        feet = {
            "FL": world[:, 0].astype(np.float32),
            "FR": world[:, 1].astype(np.float32),
            "RL": world[:, 2].astype(np.float32),
            "RR": world[:, 3].astype(np.float32),
        }
        mode_out = np.where(mode2 == 0, MODE_DS_FL_RR, MODE_DS_FR_RL).astype(np.int32)
        return body, psi, feet, mode_out

    if s.shape[1] == 16:
        # Corridor body [s, ey] + fixed nominal template + residual (matches env._decode_feet_np).
        cy = float(centerline_y)
        body = np.stack([s[:, 0], cy + s[:, 1]], axis=-1).astype(np.float32)
        psi = s[:, 2].astype(np.float32)
        res = s[:, 6:14].reshape(-1, 4, 2).astype(np.float32)
        mode2 = np.mod(np.rint(s[:, 14]).astype(np.int32), 2)

        def _pick(name: str, kw: Optional[float], default: float) -> float:
            if kw is not None:
                return float(kw)
            if env is not None and hasattr(env, name):
                return float(getattr(env, name))
            return float(default)

        x_f = _pick("x_f_nominal", x_f_nominal, 0.18)
        x_r = _pick("x_r_nominal", x_r_nominal, -0.18)
        y_L = _pick("y_L_nominal", y_L_nominal, 0.15)
        y_R = _pick("y_R_nominal", y_R_nominal, -0.15)

        c = np.cos(psi).astype(np.float32)
        sn = np.sin(psi).astype(np.float32)
        local_fl = np.stack([x_f + res[:, 0, 0], y_L + res[:, 0, 1]], axis=-1)
        local_fr = np.stack([x_f + res[:, 1, 0], y_R + res[:, 1, 1]], axis=-1)
        local_rl = np.stack([x_r + res[:, 2, 0], y_L + res[:, 2, 1]], axis=-1)
        local_rr = np.stack([x_r + res[:, 3, 0], y_R + res[:, 3, 1]], axis=-1)
        loc = np.stack([local_fl, local_fr, local_rl, local_rr], axis=1)
        vx = loc[..., 0]
        vy = loc[..., 1]
        wx = c[:, None] * vx - sn[:, None] * vy
        wy = sn[:, None] * vx + c[:, None] * vy
        world = body[:, None, :] + np.stack([wx, wy], axis=-1)
        feet = {
            "FL": world[:, 0].astype(np.float32),
            "FR": world[:, 1].astype(np.float32),
            "RL": world[:, 2].astype(np.float32),
            "RR": world[:, 3].astype(np.float32),
        }
        mode_out = np.where(mode2 == 0, MODE_DS_FL_RR, MODE_DS_FR_RL).astype(np.int32)
        return body, psi, feet, mode_out

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
            "support_platforms": np.asarray(
                getattr(scene, "support_platforms", np.zeros((0, 4), dtype=np.float32)),
                dtype=np.float32,
            ).tolist(),
        }
    except Exception:
        return None
