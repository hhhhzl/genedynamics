"""ContinuousBaseGovernor: governs a humanoid base reference (an :class:`Intent`)
into an execution-compatible reference, then writes the governed pose + velocity
back into the Intent.

Step-2 of the reference-governor theory (feedforward by default; optional
closed-loop error throttle). The tracking policy (spark, model_6000, ...) is
unchanged: it consumes the governed ``base_lin_vel`` / ``base_yaw_rate`` as
before. All projection math lives in :mod:`core`; the task-specific safety set
is injected as an :class:`AdmissibleSet`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

import numpy as np

from genedynamics.deploy.interfaces.messages import Intent, RobotState

from . import core
from .base import AdmissibleSet, GovernorDiagnostics, ReferenceGovernor
from .config import GovernorConfig


class ContinuousBaseGovernor(ReferenceGovernor):
    def __init__(self, admissible: AdmissibleSet, cfg: Optional[GovernorConfig] = None):
        self.adm = admissible
        self.cfg = cfg or GovernorConfig()
        self._r_prev_world: Optional[np.ndarray] = None  # [x, y, yaw]
        self.last_diag: Optional[GovernorDiagnostics] = None

    # -- ReferenceGovernor ----------------------------------------------------
    def reset(self, state: Any = None) -> None:
        self._r_prev_world = self._measured_pose(state)
        # Let an obstacle-aware admissible set clear any per-run state (e.g. the
        # committed pass side in BodySdfAdmissibleSet).
        adm_reset = getattr(self.adm, "reset", None)
        if callable(adm_reset):
            adm_reset(state)

    @property
    def governed_pose(self):
        """Last governed world pose ``[x, y, yaw]`` (None before first govern())."""
        return None if self._r_prev_world is None else self._r_prev_world.copy()

    def set_anchor(self, x: float, y: float, yaw: float) -> None:
        """Re-anchor the projection at a known world pose.

        Use this each step when an outer loop already tracks position (e.g. a
        follower's xy P-controller): the governor then acts as a pure
        per-step admissibility / rate / posture limiter on the commanded step,
        rather than integrating its own reference.
        """
        self._r_prev_world = np.array([float(x), float(y), float(yaw)])

    def govern(self, ref: Intent, state: Optional[RobotState] = None, ctx: Any = None) -> Intent:
        cfg = self.cfg
        r_ref_w = self._ref_from_intent(ref)
        x_meas = self._measured_pose(state)
        if self._r_prev_world is None:
            self._r_prev_world = x_meas.copy() if x_meas is not None else r_ref_w.copy()
        r_prev_w = self._r_prev_world

        frame = self.adm.reference_frame(ctx)
        r_ref_f = core.to_frame(r_ref_w[:2], r_ref_w[2], frame.origin, frame.yaw)
        r_prev_f = core.to_frame(r_prev_w[:2], r_prev_w[2], frame.origin, frame.yaw)

        if x_meas is not None:
            x_meas_f = core.to_frame(x_meas[:2], x_meas[2], frame.origin, frame.yaw)
            err = float(np.hypot(*(x_meas_f[:2] - r_prev_f[:2])))
        else:
            err = 0.0
        thr = (
            core.throttle_factor(err, cfg.error_deadband, cfg.error_throttle_gain)
            if cfg.closed_loop
            else 1.0
        )

        # Plan step in delta space (relative to previous governed reference).
        d_ref = np.array(
            [
                r_ref_f[0] - r_prev_f[0],
                r_ref_f[1] - r_prev_f[1],
                core.wrap_angle(r_ref_f[2] - r_prev_f[2]),
            ]
        )

        # --- boxes (delta space) ------------------------------------------
        rate_lo = thr * np.array([cfg.v_min_lon * cfg.dt, -cfg.v_max_lat * cfg.dt, -cfg.yaw_rate_max * cfg.dt])
        rate_hi = thr * np.array([cfg.v_max_lon * cfg.dt, cfg.v_max_lat * cfg.dt, cfg.yaw_rate_max * cfg.dt])

        if cfg.pelvis_roll_gain > 1e-9:
            dn_post = max(0.0, cfg.phi_max - cfg.m_phi) / cfg.pelvis_roll_gain * cfg.dt
        else:
            dn_post = np.inf
        post_lo = np.array([-np.inf, -dn_post, -np.inf])
        post_hi = np.array([np.inf, dn_post, np.inf])

        sbox = self.adm.safety_box(r_ref_f, ctx, cfg)  # absolute frame bounds
        saf_lo = np.where(np.isfinite(sbox.lo), sbox.lo - r_prev_f, -np.inf)
        saf_hi = np.where(np.isfinite(sbox.hi), sbox.hi - r_prev_f, np.inf)

        lo, hi = core.intersect_boxes([rate_lo, post_lo, saf_lo], [rate_hi, post_hi, saf_hi])
        infeasible = bool(np.any(lo > hi + 1e-9))
        if infeasible:
            # Safety wins; fall back to rate bounds only where safety is inactive.
            lo = np.where(np.isfinite(saf_lo), saf_lo, rate_lo)
            hi = np.where(np.isfinite(saf_hi), saf_hi, rate_hi)
            lo = np.minimum(lo, hi)  # guarantee a non-empty (degenerate) box

        # --- optional obstacle-aware safety half-space (a . r >= b) -------
        # The half-space is expressed in absolute frame coords; convert to
        # delta space about r_prev_f: a . (r_prev + d) >= b => a . d >= b'.
        hspace = self.adm.safety_halfspace(r_ref_f, ctx, cfg)
        halfspace_active = False

        # --- proximal + clip (+ half-space projection) --------------------
        q = np.array([cfg.q_pos, cfg.q_pos, cfg.q_yaw])
        d_prox = core.proximal_delta(d_ref, q, cfg.eta)
        if getattr(hspace, "active", False):
            a_hs = np.asarray(hspace.a, dtype=float)
            b_hs = float(hspace.b) - float(np.dot(a_hs, r_prev_f))
            # The half-space must NOT be sacrificed to the rate box: when the two
            # conflict (a fast push needed to clear an obstacle), favour safety
            # but stay inside the rate box so the policy still receives a
            # trackable command. project_box_halfspace clips into the box and
            # pushes along +a; if box-saturated it returns the box-feasible point
            # that maximises a.d (= maximally safe).
            d_gov, halfspace_active = core.project_box_halfspace(d_prox, lo, hi, a_hs, b_hs)
        else:
            d_gov = core.clip(d_prox, lo, hi)

        r_gov_f = np.array(
            [r_prev_f[0] + d_gov[0], r_prev_f[1] + d_gov[1], core.wrap_angle(r_prev_f[2] + d_gov[2])]
        )
        gov_xy, gov_yaw = core.from_frame(r_gov_f, frame.origin, frame.yaw)
        r_gov_w = np.array([gov_xy[0], gov_xy[1], gov_yaw])

        # --- write governed pose + (body-frame) velocity back into Intent --
        dxy_w = r_gov_w[:2] - r_prev_w[:2]
        yaw_prev = r_prev_w[2]
        c, s = np.cos(-yaw_prev), np.sin(-yaw_prev)
        v_body = np.array(
            [(c * dxy_w[0] - s * dxy_w[1]) / cfg.dt, (s * dxy_w[0] + c * dxy_w[1]) / cfg.dt]
        )
        yaw_rate = core.wrap_angle(r_gov_w[2] - yaw_prev) / cfg.dt

        governed = replace(
            ref,
            base_pos_xy=r_gov_w[:2].copy(),
            base_yaw=float(r_gov_w[2]),
            base_lin_vel=v_body,
            base_yaw_rate=float(yaw_rate),
        )

        self.last_diag = GovernorDiagnostics(
            governed=r_gov_w,
            reference=r_ref_w,
            tracking_error=err,
            rate_active=bool(
                np.any(np.isclose(d_gov, rate_lo)) or np.any(np.isclose(d_gov, rate_hi))
            ),
            safety_active=bool(np.isfinite(sbox.hi[1]) or halfspace_active),
            posture_active=bool(np.isfinite(dn_post)),
            infeasible=infeasible,
            throttle=thr,
            extras={"halfspace_active": halfspace_active},
        )
        self._r_prev_world = r_gov_w.copy()
        return governed

    # -- helpers --------------------------------------------------------------
    def _ref_from_intent(self, intent: Intent) -> np.ndarray:
        if intent.base_pos_xy is not None:
            xy = np.asarray(intent.base_pos_xy, dtype=float)
        elif self._r_prev_world is not None:
            yaw = self._r_prev_world[2]
            v = np.asarray(intent.base_lin_vel, dtype=float)
            rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            xy = self._r_prev_world[:2] + rot @ v * self.cfg.dt
        else:
            xy = np.zeros(2)
        return np.array([xy[0], xy[1], float(intent.base_yaw)])

    @staticmethod
    def _measured_pose(state: Any) -> Optional[np.ndarray]:
        if state is None or getattr(state, "base_pose", None) is None:
            return None
        bp = np.asarray(state.base_pose, dtype=float)
        x, y = bp[0], bp[1]
        w, xq, yq, zq = bp[3], bp[4], bp[5], bp[6]
        yaw = np.arctan2(2.0 * (w * zq + xq * yq), 1.0 - 2.0 * (yq * yq + zq * zq))
        return np.array([x, y, yaw])
