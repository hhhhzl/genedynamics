"""
Stepping reference governor (Rec. 2 + 3).

Sits between the planner and the walker.  The planner emits a *continuous*
body-path + fixed foot-template (all four feet ride the body forward together; there
is no real swing/stance separation and the contact-mode label barely switches).  The
current walker groups long repeated-mode runs into one execution interval and tries to
realize it as a single multi-metre swing -> ``consecutive_touchdown_timeouts``.

The governor turns the planner's body-path into an **explicit discrete gait reference
with support transitions**:

1. Extract and arclength-resample the planner body-path (mid, yaw).
2. Synthesize a **single-support walk** gait (one leg swings at a time; the other three
   stay planted as fixed world anchors).  Body advance per cycle ~= one stone pitch, so
   every swing is hard-bounded at ``l_max``.  Three contacts are held at all times so the
   walker's active base-PD always has support to push against; the CoM is not required to
   stay strictly inside the 3-foot triangle (``goal_in_support`` reports when it does).
3. Project every target foothold into the safe **interior** of the nearest stone, so
   footholds are physical contact anchors (this also fixes the stance-drift problem,
   because stance anchors are frozen during a step instead of re-blended to the plan).
4. Rank candidate trajectories and pick the one that produces the cleanest executable
   gait reference.

The low-level walker is unchanged; it consumes the :class:`GaitReference` through a
clean hook (``SteppingWalkFollowerMinimal.follow_gait_reference``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from genedynamics.tasks.stepping_stones import decode_plan_states

LEG_ORDER = ("FL", "FR", "RL", "RR")

# Statically-supported crawl order: alternates side and end (diagonal ABAB crawl).
# Each leg swings exactly once per cycle; the other three remain planted.
DEFAULT_WALK_ORDER: Tuple[str, ...] = ("FL", "RR", "FR", "RL")
# Diagonal trot pairs (kept for completeness; default gait is "walk").
TROT_GROUPS: Tuple[Tuple[str, ...], ...] = (("FL", "RR"), ("FR", "RL"))


# ============================================================================
# Execution-aware planner diagnostics (Rec. 1)
#
# The task-level planner metric (``stepping_metrics``) only checks *adjacent* states, so
# a continuous-residual plan looks locally feasible.  The walker groups contiguous equal
# contact modes into one execution interval and tries to realize each as one physical
# swing; a long repeated-mode run collapses into one multi-metre swing no walker can latch
# (-> ``consecutive_touchdown_timeouts``).  These diagnostics report the quantities that
# predict walker-compatibility and expose them as *gates*, so the governor can rank/reject
# a candidate before it is handed to the simulator.  The interval grouping mirrors
# ``SteppingWalkFollowerMinimal._mode_intervals`` / ``_swing_legs_from_mode`` exactly.
# ============================================================================
def _swing_legs_from_mode(mode: int) -> Tuple[str, ...]:
    """Mirror of ``stepping_walker._swing_legs_from_mode`` (walker reads raw col-14)."""
    m = int(mode) % 4
    if m in (0, 1):  # MODE_DS_FL_RR / MODE_QS_AFTER_FL_RR
        return ("FR", "RL")
    return ("FL", "RR")


def _stance_legs_from_mode(mode: int) -> Tuple[str, ...]:
    swing = set(_swing_legs_from_mode(mode))
    return tuple(leg for leg in LEG_ORDER if leg not in swing)


def mode_intervals(modes: Sequence[int], n_seg: int) -> List[Tuple[int, int, int]]:
    """Group contiguous equal modes into (seg_start, seg_end, mode) intervals.

    Exact mirror of ``SteppingWalkFollowerMinimal._mode_intervals``.
    """
    if n_seg <= 0:
        return []
    mm = np.asarray(modes, dtype=np.int32).reshape(-1)
    if mm.size < n_seg:
        mm = np.pad(mm, (0, n_seg - mm.size), mode="edge")
    out: List[Tuple[int, int, int]] = []
    start = 0
    curr = int(mm[0])
    for k in range(1, n_seg):
        mk = int(mm[k])
        if mk != curr:
            out.append((start, k - 1, curr))
            start = k
            curr = mk
    out.append((start, n_seg - 1, curr))
    return out


@dataclass
class ExecGates:
    """Thresholds that decide whether a plan is walker-executable.

    ``swing_step_limit`` defaults to the scene ``l_max`` (0.35 for level 1).  A plan
    interval whose swing exceeds it cannot be one physical step.  ``ratio_max`` adds a
    safety factor on top (1.0 = exactly at the kinematic limit).
    """

    swing_step_limit: float = 0.35
    body_step_limit: float = 0.22
    ratio_max: float = 1.15
    stance_drift_max: float = 0.06


@dataclass
class ExecDiagnostics:
    """Per-candidate execution-aware diagnostics + gate verdict."""

    n_states: int
    n_intervals: int
    mode_sequence: List[int]
    # adjacent-state view (what the planner metric sees)
    adjacent_swing_max: float
    adjacent_body_max: float
    # mode-interval view (what the current walker executes)
    mode_interval_swing_max: float
    mode_interval_swing_ratio: float
    mode_interval_body_max: float
    stance_drift_max: float
    stance_drift_mean: float
    per_interval: List[Dict[str, Any]] = field(default_factory=list)
    walker_compatible: bool = False
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def _decode_plan_modes(states: np.ndarray, *, step_width: float, centerline_y: float,
                       x_f: float, x_r: float, y_L: float, y_R: float, half_pair_length: float):
    body, yaw, feet, _mode = decode_plan_states(
        states, step_width=float(step_width), half_pair_length=float(half_pair_length),
        centerline_y=float(centerline_y),
        x_f_nominal=float(x_f), x_r_nominal=float(x_r),
        y_L_nominal=float(y_L), y_R_nominal=float(y_R),
    )
    st = np.asarray(states, dtype=np.float32)
    # Walker reads the *raw* mode column (col 14 for >=16D, col 10 for 12D), mod 4.
    n = body.shape[0]
    modes = np.zeros((n,), dtype=np.int32)
    if st.ndim == 2 and st.shape[0] == n:
        if st.shape[1] >= 16:
            modes = np.mod(np.rint(st[:, 14]).astype(np.int32), 4)
        elif st.shape[1] >= 12:
            modes = np.mod(np.rint(st[:, 10]).astype(np.int32), 4)
    return body, yaw, feet, modes


def diagnose_plan(
    states: np.ndarray,
    *,
    gates: Optional[ExecGates] = None,
    step_width: float = 0.30,
    centerline_y: float = 0.0,
    x_f_nominal: float = 0.18,
    x_r_nominal: float = -0.18,
    y_L_nominal: float = 0.15,
    y_R_nominal: float = -0.15,
    half_pair_length: float = 0.18,
) -> ExecDiagnostics:
    """Compute execution-aware diagnostics for a single (T, D) plan-state array."""
    gates = gates or ExecGates()
    body, yaw, feet, modes = _decode_plan_modes(
        states, step_width=step_width, centerline_y=centerline_y,
        x_f=x_f_nominal, x_r=x_r_nominal, y_L=y_L_nominal, y_R=y_R_nominal,
        half_pair_length=half_pair_length,
    )
    n = int(body.shape[0])
    n_seg = max(0, n - 1)

    # adjacent-state view (planner-metric perspective)
    adj_swing = 0.0
    adj_body = 0.0
    for t in range(n_seg):
        sw = _swing_legs_from_mode(int(modes[t]))
        if sw:
            adj_swing = max(adj_swing, max(float(np.linalg.norm(feet[lg][t + 1] - feet[lg][t])) for lg in sw))
        adj_body = max(adj_body, float(np.linalg.norm(body[t + 1] - body[t])))

    # mode-interval view (walker-execution perspective)
    intervals = mode_intervals(modes, n_seg)
    per_interval: List[Dict[str, Any]] = []
    iv_swing_max = 0.0
    iv_body_max = 0.0
    for idx, (s0, s1, mk) in enumerate(intervals):
        sw = _swing_legs_from_mode(int(mk))
        body_move = float(np.linalg.norm(body[s1 + 1] - body[s0]))
        swing_move = 0.0
        per_leg: Dict[str, float] = {}
        for lg in sw:
            dd = float(np.linalg.norm(feet[lg][s1 + 1] - feet[lg][s0]))
            per_leg[lg] = dd
            swing_move = max(swing_move, dd)
        iv_swing_max = max(iv_swing_max, swing_move)
        iv_body_max = max(iv_body_max, body_move)
        per_interval.append({
            "interval": int(idx), "seg_start": int(s0), "seg_end": int(s1), "mode": int(mk),
            "n_states": int(s1 - s0 + 1), "swing_legs": list(sw), "swing_move": swing_move,
            "swing_ratio": float(swing_move / max(gates.swing_step_limit, 1e-6)),
            "body_move": body_move, "per_leg_swing": per_leg,
        })

    # stance drift (per adjacent step, max over that step's stance legs)
    drift = []
    for t in range(n_seg):
        stance = _stance_legs_from_mode(int(modes[t]))
        if stance:
            drift.append(max(float(np.linalg.norm(feet[lg][t + 1] - feet[lg][t])) for lg in stance))
    drift_arr = np.asarray(drift, dtype=np.float64) if drift else np.zeros((0,), dtype=np.float64)
    stance_drift_max = float(np.max(drift_arr)) if drift_arr.size else 0.0
    stance_drift_mean = float(np.mean(drift_arr)) if drift_arr.size else 0.0
    swing_ratio = float(iv_swing_max / max(gates.swing_step_limit, 1e-6))

    reasons: List[str] = []
    if iv_swing_max > gates.swing_step_limit * gates.ratio_max:
        reasons.append(f"mode_interval_swing {iv_swing_max:.3f}m exceeds "
                       f"{gates.swing_step_limit * gates.ratio_max:.3f}m ({swing_ratio:.1f}x l_max)")
    if iv_body_max > gates.body_step_limit * gates.ratio_max:
        reasons.append(f"mode_interval_body {iv_body_max:.3f}m exceeds "
                       f"{gates.body_step_limit * gates.ratio_max:.3f}m")
    if stance_drift_max > gates.stance_drift_max:
        reasons.append(f"stance_drift {stance_drift_max:.3f}m exceeds {gates.stance_drift_max:.3f}m")

    return ExecDiagnostics(
        n_states=n, n_intervals=len(intervals), mode_sequence=[int(m) for m in modes.tolist()],
        adjacent_swing_max=adj_swing, adjacent_body_max=adj_body,
        mode_interval_swing_max=iv_swing_max, mode_interval_swing_ratio=swing_ratio,
        mode_interval_body_max=iv_body_max, stance_drift_max=stance_drift_max,
        stance_drift_mean=stance_drift_mean, per_interval=per_interval,
        walker_compatible=(len(reasons) == 0), reasons=reasons,
    )


def diagnose_candidates(
    candidate_states: Sequence[np.ndarray],
    *,
    gates: Optional[ExecGates] = None,
    **decode_kwargs: Any,
) -> List[ExecDiagnostics]:
    """Diagnose every candidate trajectory in a planner result."""
    return [diagnose_plan(np.asarray(c, dtype=np.float32), gates=gates, **decode_kwargs)
            for c in candidate_states]


def _wrap_angle(x: float) -> float:
    return float(np.arctan2(np.sin(x), np.cos(x)))


def _interp_angle(a0: float, a1: float, alpha: float) -> float:
    return float(a0 + float(alpha) * _wrap_angle(a1 - a0))


@dataclass
class GovernorConfig:
    gait: str = "walk"  # walk | trot
    # Template (must match env / walker MinimalFollowerConfig).
    x_f_nominal: float = 0.18
    x_r_nominal: float = -0.18
    y_L_nominal: float = 0.15
    y_R_nominal: float = -0.15
    step_width: float = 0.30
    half_pair_length: float = 0.18
    centerline_y: float = 0.0
    # Gait timing / geometry.
    stride_length: float = 0.22       # body advance per full gait cycle (~ stone pitch)
    walk_order: Tuple[str, ...] = DEFAULT_WALK_ORDER
    swing_step_limit: float = 0.35    # l_max; hard bound on a single swing
    # Foothold projection.
    project_to_stones: bool = True
    stone_interior_clearance: float = 0.025   # keep foot this far inside the stone rim
    min_foot_z: float = 0.015
    stone_top_z: float = 0.035
    # Candidate ranking gates.
    rank_gates: ExecGates = field(default_factory=ExecGates)


@dataclass
class StepPhase:
    """One single-leg (walk) or diagonal-pair (trot) execution step."""

    index: int
    swing_legs: Tuple[str, ...]
    stance_legs: Tuple[str, ...]
    body_mid_start: np.ndarray
    body_mid_goal: np.ndarray
    yaw_start: float
    yaw_goal: float
    foot_goal: Dict[str, np.ndarray]   # world xyz target for every leg this step
    swing_distance: float
    swing_on_stone: bool
    goal_in_support: bool              # diagnostic: body goal inside the 3-foot support polygon


@dataclass
class GaitReference:
    phases: List[StepPhase]
    gait: str
    body_path: np.ndarray              # (M+1, 2) resampled body mids (for logging/replay)
    yaw_path: np.ndarray               # (M+1,)
    source_candidate_idx: int
    max_swing_distance: float
    all_on_stone: bool
    n_steps: int
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return len(self.phases) == 0


class SteppingReferenceGovernor:
    def __init__(self, cfg: Optional[GovernorConfig] = None,
                 stepping_scene: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = cfg or GovernorConfig()
        self.scene = stepping_scene or {}
        centers = np.asarray(self.scene.get("stones_centers", []), dtype=np.float64).reshape(-1, 2)
        radii = np.asarray(self.scene.get("stones_radii", []), dtype=np.float64).reshape(-1)
        n = min(centers.shape[0], radii.shape[0])
        self._centers = centers[:n]
        self._radii = radii[:n]

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def _template_local(self) -> Dict[str, np.ndarray]:
        c = self.cfg
        return {
            "FL": np.array([c.x_f_nominal, c.y_L_nominal], dtype=np.float64),
            "FR": np.array([c.x_f_nominal, c.y_R_nominal], dtype=np.float64),
            "RL": np.array([c.x_r_nominal, c.y_L_nominal], dtype=np.float64),
            "RR": np.array([c.x_r_nominal, c.y_R_nominal], dtype=np.float64),
        }

    def _nominal_feet(self, body_xy: np.ndarray, yaw: float) -> Dict[str, np.ndarray]:
        tpl = self._template_local()
        c, s = float(np.cos(yaw)), float(np.sin(yaw))
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        b = np.asarray(body_xy, dtype=np.float64).reshape(2)
        return {leg: (b + R @ tpl[leg]).astype(np.float64) for leg in LEG_ORDER}

    def _terrain_z(self, xy: np.ndarray) -> float:
        if self._centers.shape[0] == 0:
            return 0.0
        d2 = np.sum((self._centers - np.asarray(xy, dtype=np.float64).reshape(1, 2)) ** 2, axis=1)
        i = int(np.argmin(d2))
        if d2[i] <= float(self._radii[i]) ** 2:
            return float(self.cfg.stone_top_z)
        return 0.0

    def _project_to_stone(
        self,
        xy: np.ndarray,
        *,
        reach_from: Optional[np.ndarray] = None,
        reach_limit: Optional[float] = None,
    ) -> Tuple[np.ndarray, bool, int]:
        """Project a 2D target into the interior of the best supporting stone.

        Returns ``(xy_proj, on_stone, stone_idx)``.  ``on_stone`` is True when the result
        lies inside a stone with the requested interior clearance.  When ``reach_from`` /
        ``reach_limit`` are given, only stones whose nearest interior point lies within
        ``reach_limit`` of ``reach_from`` are eligible (so the snap cannot select a stone
        the foot could not actually swing to within one step); if none qualify the call
        returns ``(xy, False, -1)``.
        """
        p = np.asarray(xy, dtype=np.float64).reshape(2)
        if not self.cfg.project_to_stones or self._centers.shape[0] == 0:
            return p, False, -1
        clr = float(self.cfg.stone_interior_clearance)
        r_eff_all = np.maximum(0.0, self._radii - clr)
        delta = p[None, :] - self._centers
        dist = np.linalg.norm(delta, axis=1)
        interior = self._radii - dist                       # >0 means inside stone
        if reach_from is not None and reach_limit is not None:
            rf = np.asarray(reach_from, dtype=np.float64).reshape(2)
            d_center = np.linalg.norm(self._centers - rf[None, :], axis=1)
            reachable = (d_center - r_eff_all) <= float(reach_limit) + 1e-9
            if not np.any(reachable):
                return p, False, -1
            interior = np.where(reachable, interior, -np.inf)
        best = int(np.argmax(interior))
        if not np.isfinite(interior[best]):
            return p, False, -1
        r_eff = float(r_eff_all[best])
        if float(dist[best]) <= r_eff:
            return p, True, best                            # already comfortably inside
        # snap onto the interior circle of the chosen stone
        direction = delta[best]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            return self._centers[best].copy(), True, best
        proj = self._centers[best] + direction / norm * r_eff
        on_stone = r_eff > 1e-6
        return proj.astype(np.float64), bool(on_stone), best

    def _to_world3(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64).reshape(2)
        z = max(float(self.cfg.min_foot_z), float(self._terrain_z(xy)))
        return np.array([xy[0], xy[1], z], dtype=np.float64)

    # ------------------------------------------------------------------
    # Body-path extraction & resampling
    # ------------------------------------------------------------------
    def _decode_body_path(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        c = self.cfg
        body, yaw, _feet, _mode = decode_plan_states(
            states, step_width=c.step_width, half_pair_length=c.half_pair_length,
            centerline_y=c.centerline_y,
            x_f_nominal=c.x_f_nominal, x_r_nominal=c.x_r_nominal,
            y_L_nominal=c.y_L_nominal, y_R_nominal=c.y_R_nominal,
        )
        return np.asarray(body, dtype=np.float64), np.asarray(yaw, dtype=np.float64).reshape(-1)

    @staticmethod
    def _arclen(mids: np.ndarray) -> np.ndarray:
        if mids.shape[0] <= 1:
            return np.zeros((mids.shape[0],), dtype=np.float64)
        d = np.linalg.norm(mids[1:] - mids[:-1], axis=1)
        return np.concatenate([[0.0], np.cumsum(d)]).astype(np.float64)

    def _resample(self, mids: np.ndarray, yaws: np.ndarray, n_out: int) -> Tuple[np.ndarray, np.ndarray]:
        s = self._arclen(mids)
        L = float(s[-1]) if s.size else 0.0
        out_mid = np.zeros((n_out + 1, 2), dtype=np.float64)
        out_yaw = np.zeros((n_out + 1,), dtype=np.float64)
        if mids.shape[0] <= 1 or L <= 1e-9:
            out_mid[:] = mids[0]
            out_yaw[:] = yaws[0] if yaws.size else 0.0
            return out_mid, out_yaw
        for j in range(n_out + 1):
            sq = (j / float(n_out)) * L
            idx = int(np.searchsorted(s, sq, side="right") - 1)
            idx = max(0, min(idx, mids.shape[0] - 2))
            seg = float(s[idx + 1] - s[idx])
            frac = 0.0 if seg <= 1e-9 else float((sq - s[idx]) / seg)
            out_mid[j] = (1.0 - frac) * mids[idx] + frac * mids[idx + 1]
            out_yaw[j] = _interp_angle(float(yaws[idx]), float(yaws[idx + 1]), frac)
        return out_mid, out_yaw

    # ------------------------------------------------------------------
    # Gait synthesis
    # ------------------------------------------------------------------
    def synthesize(self, states: np.ndarray, *, source_idx: int = -1) -> GaitReference:
        """Synthesize a walk/trot gait reference from a single plan-state array."""
        cfg = self.cfg
        mids, yaws = self._decode_body_path(states)
        if mids.shape[0] == 0:
            return GaitReference([], cfg.gait, mids, yaws, source_idx, 0.0, True, 0)

        L = float(self._arclen(mids)[-1])
        groups: Tuple[Tuple[str, ...], ...]
        if cfg.gait == "trot":
            groups = TROT_GROUPS
        else:
            groups = tuple((leg,) for leg in cfg.walk_order)
        cyc = len(groups)
        n_cycles = max(1, int(round(L / max(cfg.stride_length, 1e-6))))
        n_steps = n_cycles * cyc
        poses_mid, poses_yaw = self._resample(mids, yaws, n_steps)

        # init planted footholds from projected nominal feet at the first pose
        planted: Dict[str, np.ndarray] = {}
        for leg in LEG_ORDER:
            nf = self._nominal_feet(poses_mid[0], float(poses_yaw[0]))[leg]
            proj, _on, _i = self._project_to_stone(nf)
            planted[leg] = self._to_world3(proj)

        phases: List[StepPhase] = []
        max_swing = 0.0
        all_on_stone = True
        n_stalled = 0
        for j in range(n_steps):
            swing = groups[j % cyc]
            stance = tuple(leg for leg in LEG_ORDER if leg not in swing)
            mid0, mid1 = poses_mid[j], poses_mid[j + 1]
            yaw0, yaw1 = float(poses_yaw[j]), float(poses_yaw[j + 1])
            nominal_next = self._nominal_feet(mid1, yaw1)

            foot_goal: Dict[str, np.ndarray] = {leg: planted[leg].copy() for leg in stance}
            step_swing = 0.0
            step_on_stone = True
            for leg in swing:
                proj, on_stone, _i = self._project_to_stone(nominal_next[leg])
                target = self._to_world3(proj)
                d = float(np.linalg.norm(target[:2] - planted[leg][:2]))
                if d > cfg.swing_step_limit:
                    # Nominal foothold is beyond one swing: snap to the best stone actually
                    # reachable within l_max of the planted foot that makes real progress
                    # toward the nominal.  ``swing_step_limit`` is a HARD bound -- if no such
                    # stone exists, do NOT advance the foot over-limit; keep it planted this
                    # step and flag the stall (the body outran foot reach -> infeasible).
                    nom_xy = np.asarray(nominal_next[leg], dtype=np.float64).reshape(2)
                    proj2, on2, _i2 = self._project_to_stone(
                        nominal_next[leg], reach_from=planted[leg][:2], reach_limit=cfg.swing_step_limit)
                    advanced = False
                    if on2:
                        t2 = self._to_world3(proj2)
                        d2 = float(np.linalg.norm(t2[:2] - planted[leg][:2]))
                        prog = (float(np.linalg.norm(planted[leg][:2] - nom_xy))
                                - float(np.linalg.norm(t2[:2] - nom_xy)))
                        if d2 <= cfg.swing_step_limit + 1e-6 and prog > 1e-3:
                            target, d, on_stone, advanced = t2, d2, True, True
                    if not advanced:
                        target, d, on_stone = planted[leg].copy(), 0.0, False
                        n_stalled += 1
                foot_goal[leg] = target
                step_swing = max(step_swing, d)
                step_on_stone = step_on_stone and bool(on_stone)
                planted[leg] = target
            max_swing = max(max_swing, step_swing)
            all_on_stone = all_on_stone and step_on_stone

            # Diagnostic: does the (fully-advanced) body goal lie in the 3-foot support
            # polygon?  Reported, not enforced -- the walker's active base PD provides the
            # stabilizing wrench, so the CoM is not required to stay strictly inside.
            goal_in_support = self._goal_in_support(mid1, [foot_goal[leg][:2] for leg in stance])
            phases.append(StepPhase(
                index=j, swing_legs=swing, stance_legs=stance,
                body_mid_start=mid0.copy(), body_mid_goal=mid1.copy(),
                yaw_start=yaw0, yaw_goal=yaw1,
                foot_goal={leg: foot_goal[leg].copy() for leg in LEG_ORDER},
                swing_distance=step_swing, swing_on_stone=step_on_stone,
                goal_in_support=goal_in_support,
            ))

        diagnostics = {
            "n_cycles": int(n_cycles), "n_steps": int(n_steps),
            "body_path_length": L,
            "max_swing_distance": float(max_swing),
            "swing_ratio": float(max_swing / max(cfg.swing_step_limit, 1e-6)),
            "all_on_stone": bool(all_on_stone),
            "n_stalled_steps": int(n_stalled),
            "feasible": bool(n_stalled == 0 and max_swing <= cfg.swing_step_limit + 1e-6),
            "goal_in_support_ratio": float(np.mean([1.0 if p.goal_in_support else 0.0 for p in phases])) if phases else 1.0,
            "final_body_goal_error": float(np.linalg.norm(poses_mid[-1] - mids[-1])),
        }
        return GaitReference(
            phases=phases, gait=cfg.gait, body_path=poses_mid, yaw_path=poses_yaw,
            source_candidate_idx=int(source_idx), max_swing_distance=float(max_swing),
            all_on_stone=bool(all_on_stone), n_steps=int(n_steps), diagnostics=diagnostics,
        )

    def _goal_in_support(self, body_xy: np.ndarray, stance_xy: Sequence[np.ndarray]) -> bool:
        """True if body_xy lies inside (or on) the convex hull of the stance feet."""
        pts = [np.asarray(p, dtype=np.float64).reshape(2) for p in stance_xy]
        if len(pts) < 3:
            return False
        b = np.asarray(body_xy, dtype=np.float64).reshape(2)
        # sign-of-cross-product test against each triangle edge (stance feet ordered by angle)
        c = np.mean(pts, axis=0)
        pts_sorted = sorted(pts, key=lambda p: float(np.arctan2(p[1] - c[1], p[0] - c[0])))
        sign = 0
        m = len(pts_sorted)
        for i in range(m):
            a = pts_sorted[i]
            bb = pts_sorted[(i + 1) % m]
            cross = (bb[0] - a[0]) * (b[1] - a[1]) - (bb[1] - a[1]) * (b[0] - a[0])
            s = 1 if cross > 1e-9 else (-1 if cross < -1e-9 else 0)
            if s != 0:
                if sign == 0:
                    sign = s
                elif s != sign:
                    return False
        return True

    # ------------------------------------------------------------------
    # Candidate ranking
    # ------------------------------------------------------------------
    def govern(
        self,
        candidate_states: Sequence[np.ndarray],
        *,
        planner_best_idx: int = 0,
        candidate_costs: Optional[Sequence[float]] = None,
    ) -> Tuple[GaitReference, List[Dict[str, Any]]]:
        """Rank candidates and return the governed gait reference for the best one.

        Ranking prefers (in order): a gait whose every foothold lands on a stone, the
        smallest max-swing distance, then the planner's own cost / best_idx.  Because
        the governor *re-synthesizes* the gait from the body-path, every candidate
        becomes executable; ranking just picks the cleanest one.
        """
        cfg = self.cfg
        reports: List[Dict[str, Any]] = []
        refs: List[GaitReference] = []
        for i, cs in enumerate(candidate_states):
            cs = np.asarray(cs, dtype=np.float32)
            raw = diagnose_plan(
                cs, gates=cfg.rank_gates, step_width=cfg.step_width,
                centerline_y=cfg.centerline_y, x_f_nominal=cfg.x_f_nominal,
                x_r_nominal=cfg.x_r_nominal, y_L_nominal=cfg.y_L_nominal,
                y_R_nominal=cfg.y_R_nominal, half_pair_length=cfg.half_pair_length,
            )
            ref = self.synthesize(cs, source_idx=i)
            cost = float(candidate_costs[i]) if candidate_costs is not None and i < len(candidate_costs) else 0.0
            reports.append({
                "candidate": i,
                "raw_walker_compatible": raw.walker_compatible,
                "raw_mode_interval_swing_max": raw.mode_interval_swing_max,
                "raw_mode_interval_swing_ratio": raw.mode_interval_swing_ratio,
                "raw_stance_drift_max": raw.stance_drift_max,
                "governed_max_swing": ref.max_swing_distance,
                "governed_swing_ratio": ref.diagnostics.get("swing_ratio", 0.0),
                "governed_all_on_stone": ref.all_on_stone,
                "governed_feasible": ref.diagnostics.get("feasible", True),
                "governed_n_stalled_steps": ref.diagnostics.get("n_stalled_steps", 0),
                "governed_goal_in_support_ratio": ref.diagnostics.get("goal_in_support_ratio", 0.0),
                "governed_n_steps": ref.n_steps,
                "planner_cost": cost,
            })
            refs.append(ref)

        def sort_key(i: int):
            r = reports[i]
            return (
                0 if r["governed_feasible"] else 1,           # bounded + no stalls first
                0 if r["governed_all_on_stone"] else 1,       # on-stone next
                r["governed_n_stalled_steps"],                # fewest stalls
                r["governed_max_swing"],                      # smallest swing
                -(r["governed_goal_in_support_ratio"]),       # most static-support margin
                0 if i == planner_best_idx else 1,            # honor planner pick
                r["planner_cost"],                            # then planner cost
            )

        order = sorted(range(len(refs)), key=sort_key)
        best = order[0] if order else planner_best_idx
        for rank, i in enumerate(order):
            reports[i]["governed_rank"] = rank
        chosen = refs[best] if refs else GaitReference([], cfg.gait, np.zeros((0, 2)), np.zeros((0,)), -1, 0.0, True, 0)
        return chosen, reports
