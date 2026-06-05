"""Body-SDF admissible set: obstacle-aware R_adm for the humanoid corridor.

Unlike :class:`CorridorAdmissibleSet` (which models only corridor-EDGE clearance
as an axis-aligned lateral box), this set models the *actual obstacles* in the
scene and the *actual robot body* (elliptical torso + two arm tips, height-gated
against 3-D bounding boxes / spheres / quarter-circles), reusing the planner's
collision model (``HumanoidCorridor2DEnv._body_min_sdf`` / the CBF filter's
``_body_min_sdf``). It returns a LINEARISED safety half-space on the base
reference (x, y):

    SDF(r0) + grad_xy . (r - r0) >= m_track

i.e. ``a . r >= b`` with ``a = [gx, gy, 0]`` and ``b = a . r0 + m_track - SDF(r0)``.
When the body is far from every obstacle the half-space is inactive (it only
binds inside a configurable activation band), so the governor's default
feedforward + rate/posture behaviour is preserved away from obstacles.

The whole module is pure NumPy (the governor runs in the numpy deploy runtime):
the SDF and its 2-D spatial gradient (finite differences in x, y) are
self-contained so no JAX dependency leaks into deploy. The geometry constants
mirror ``genedynamics/envs/humanoid_corridor_2d.py`` and
``.../action_filters/task_specific/cbf_corridor.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .base import AdmissibleSet, Box, Frame, Halfspace

# ---------------------------------------------------------------------------
# G1 body geometry (mirrors HumanoidCorridor2DEnv / cbf_corridor).
# ---------------------------------------------------------------------------
TORSO_A = 0.20            # torso front half-width (m)
TORSO_B = 0.11            # torso side half-depth (m)
TORSO_CROUCH_EXTRA = 0.03
ARM_REACH_OPEN = 0.35
ARM_REACH_TUCKED = 0.10
ARM_RADIUS = 0.05
BODY_HALF_H = 0.25
H_NOMINAL = 0.75


@dataclass
class _Obs:
    """Parsed obstacle (box / sphere / quarter-circle)."""

    shape: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    cx: float
    cy: float
    radius: float
    qc_clip_sign: float


def _parse_obstacles(obstacles: Optional[List[Dict[str, Any]]]) -> List[_Obs]:
    out: List[_Obs] = []
    for o in obstacles or []:
        out.append(
            _Obs(
                shape=str(o.get("shape", "box")),
                x_min=float(o.get("x_min", 0.0)),
                x_max=float(o.get("x_max", 0.0)),
                y_min=float(o.get("y_min", 0.0)),
                y_max=float(o.get("y_max", 0.0)),
                z_min=float(o.get("z_min", 0.0)),
                z_max=float(o.get("z_max", 2.0)),
                cx=float(o.get("cx", 0.0)),
                cy=float(o.get("cy", 0.0)),
                radius=float(o.get("radius", 0.0)),
                qc_clip_sign=float(o.get("qc_clip_sign", 0.0)),
            )
        )
    return out


def _box_sdf(px, py, xmin, xmax, ymin, ymax):
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    hx = 0.5 * (xmax - xmin)
    hy = 0.5 * (ymax - ymin)
    dx = abs(px - cx) - hx
    dy = abs(py - cy) - hy
    outside = np.sqrt(max(dx, 0.0) ** 2 + max(dy, 0.0) ** 2 + 1e-12)
    inside = min(max(dx, dy), 0.0)
    return float(outside + inside)


def _sphere_sdf(px, py, cx, cy, r):
    return float(np.sqrt((px - cx) ** 2 + (py - cy) ** 2 + 1e-12) - r)


def _qc_sdf(px, py, cx, cy, r, clip_sign):
    circle_d = float(np.sqrt((px - cx) ** 2 + (py - cy) ** 2 + 1e-12) - r)
    x_hp = float(clip_sign * (px - cx))
    y_hp = float(py - cy) if cy > 0 else float(cy - py)
    return max(circle_d, x_hp, y_hp)


def _obs_point_sdf(px, py, o: _Obs):
    if o.shape == "sphere":
        return _sphere_sdf(px, py, o.cx, o.cy, o.radius)
    if o.shape == "qc":
        return _qc_sdf(px, py, o.cx, o.cy, o.radius, o.qc_clip_sign)
    return _box_sdf(px, py, o.x_min, o.x_max, o.y_min, o.y_max)


def _z_overlap(h, z_lo, z_hi):
    body_lo = h - BODY_HALF_H
    body_hi = h + BODY_HALF_H
    overlap = min(body_hi, z_hi) - max(body_lo, z_lo)
    return float(np.clip(overlap / (2.0 * BODY_HALF_H), 0.0, 1.0))


def _ellipse_radius(delta_phi, a, b):
    cd = np.cos(delta_phi)
    sd = np.sin(delta_phi)
    return 1.0 / np.sqrt((cd / a) ** 2 + (sd / b) ** 2 + 1e-12)


@dataclass
class BodyConfig:
    """Body posture used to place arm tips / size the torso ellipse.

    Defaults are deliberately CONSERVATIVE (arms open => widest body), so when
    the follower intent does not provide a posture the governor still keeps the
    *worst-case* body clear. Tighter (tucked) postures only relax the constraint.
    """

    h: float = H_NOMINAL          # body height (crouch grows the torso ellipse)
    psi_torso: float = 0.0        # torso yaw relative to base (rad)
    a_left: float = 0.0           # arm tuck [0 open .. 1 tucked]
    a_right: float = 0.0


@dataclass
class BodySdfContext:
    """Context for the body-SDF set: scene obstacles + corridor walls + posture.

    Built once per run from ``corridor_scene.json`` (see
    :meth:`BodySdfAdmissibleSet.from_scene_dict`). ``body`` may be refreshed each
    step from the follower intent; if absent the conservative default is used.
    """

    obstacles: List[_Obs] = field(default_factory=list)
    wall_y_min: float = -np.inf
    wall_y_max: float = np.inf
    body: BodyConfig = field(default_factory=BodyConfig)
    # The globally-consistent safe lane: the planner's own (collision-free)
    # trajectory, as a lateral offset y(x). Sampled monotonically in x. When
    # present, the half-space steers the reference toward this lane (the plan
    # already chose a feasible pass side, so this avoids the myopic-gradient
    # failure of picking the wrong side near an obstacle). Optional; if empty the
    # set falls back to the local max-clearance lateral search.
    plan_x: Optional[np.ndarray] = None  # (T,) plan pelvis x, sorted ascending
    plan_y: Optional[np.ndarray] = None  # (T,) plan pelvis y at plan_x


class BodySdfAdmissibleSet(AdmissibleSet):
    """Obstacle-aware admissible set built from the planner body SDF.

    The reference frame is the WORLD frame (identity), so the governed reference
    delta is directly ``(dx, dy, dyaw)`` in world coordinates and the half-space
    normal is the world-frame spatial SDF gradient. The lateral corridor-edge
    box is still emitted via :meth:`safety_box` (a cheap, always-on guard), and
    the obstacle half-space is added on top via :meth:`safety_halfspace`.

    Parameters
    ----------
    ctx
        A :class:`BodySdfContext` (or it can be supplied per-call via ``ctx``).
    activation_band
        The half-space only binds when ``SDF < m_track + activation_band`` so
        far-from-obstacle steps are untouched (feedforward preserved).
    grad_eps
        Finite-difference step for the 2-D spatial SDF gradient.
    """

    def __init__(
        self,
        ctx: Optional[BodySdfContext] = None,
        *,
        activation_band: float = 0.08,
        grad_eps: float = 1e-3,
        lookahead: float = 0.30,
        lag_margin: float = 0.10,
    ) -> None:
        self._ctx = ctx
        self.activation_band = float(activation_band)
        self.grad_eps = float(grad_eps)
        self.lookahead = float(lookahead)
        # Anticipatory lateral overshoot (m) added in the plan's off-centre
        # direction to pre-compensate the policy's lateral tracking lag, so the
        # *executed* body reaches the plan's safe lane instead of lagging inboard.
        self.lag_margin = float(lag_margin)
        self.last_sdf: Optional[float] = None
        self.last_grad: Optional[np.ndarray] = None
        # Committed pass side per obstacle group (hysteresis so the reference
        # doesn't dither across the obstacle midline). +1 => pass on +lateral
        # side, -1 => -lateral side. Cleared when the body is past + clear.
        self._committed_side: Optional[float] = None
        self._commit_x: Optional[float] = None

    def reset(self, state: Any = None) -> None:
        self._committed_side = None
        self._commit_x = None

    # -- construction ---------------------------------------------------------
    @classmethod
    def from_scene_dict(
        cls,
        scene: Dict[str, Any],
        *,
        body: Optional[BodyConfig] = None,
        activation_band: float = 0.08,
        lookahead: float = 0.30,
        lag_margin: float = 0.10,
        plan_xy: Optional[np.ndarray] = None,
    ) -> "BodySdfAdmissibleSet":
        obstacles = _parse_obstacles(scene.get("obstacles"))
        cw = scene.get("corridor_width")
        if cw is not None:
            wy_min, wy_max = -float(cw) / 2.0, float(cw) / 2.0
        else:
            wy_min, wy_max = -np.inf, np.inf
        px = py = None
        if plan_xy is not None:
            pxy = np.asarray(plan_xy, dtype=float).reshape(-1, 2)
            order = np.argsort(pxy[:, 0])
            px = pxy[order, 0]
            py = pxy[order, 1]
        c = BodySdfContext(
            obstacles=obstacles,
            wall_y_min=wy_min,
            wall_y_max=wy_max,
            body=body or BodyConfig(),
            plan_x=px,
            plan_y=py,
        )
        return cls(c, activation_band=activation_band, lookahead=lookahead,
                   lag_margin=lag_margin)

    @staticmethod
    def _plan_lane_y(c: "BodySdfContext", x: float) -> Optional[float]:
        """Interpolate the plan's lateral offset y at longitudinal x (or None)."""
        if c.plan_x is None or c.plan_y is None or c.plan_x.size == 0:
            return None
        return float(np.interp(float(x), c.plan_x, c.plan_y))

    def set_body(self, body: BodyConfig) -> None:
        if self._ctx is not None:
            self._ctx.body = body

    def set_plan_lane(self, plan_xy: np.ndarray) -> None:
        """Install / refresh the planner's collision-free lateral lane y(x).

        ``plan_xy`` is the planner pelvis trajectory ``(T, 2)`` in the SAME world
        frame the governor references live in (apply any warmup offset first).
        """
        if self._ctx is None:
            return
        pxy = np.asarray(plan_xy, dtype=float).reshape(-1, 2)
        if pxy.shape[0] == 0:
            self._ctx.plan_x = None
            self._ctx.plan_y = None
            return
        order = np.argsort(pxy[:, 0])
        self._ctx.plan_x = pxy[order, 0]
        self._ctx.plan_y = pxy[order, 1]

    # -- SDF ------------------------------------------------------------------
    def _resolve_ctx(self, ctx: Any) -> BodySdfContext:
        if isinstance(ctx, BodySdfContext):
            return ctx
        if self._ctx is not None:
            return self._ctx
        return BodySdfContext()

    @staticmethod
    def _arm_pos(px, py, heading, a_tuck, sign):
        reach = ARM_REACH_OPEN + (ARM_REACH_TUCKED - ARM_REACH_OPEN) * float(
            np.clip(a_tuck, 0.0, 1.0)
        )
        c, s = np.cos(heading), np.sin(heading)
        # Arm extends laterally in torso frame: local = (0, sign*reach).
        return px - s * sign * reach, py + c * sign * reach

    def body_min_sdf(self, x: float, y: float, yaw: float, c: BodySdfContext) -> float:
        """Min signed distance body -> (obstacles ∪ walls). Positive = clear."""
        b = c.body
        h = float(b.h)
        heading = float(yaw) + float(b.psi_torso)
        a_eff = TORSO_A + TORSO_CROUCH_EXTRA * max(0.0, H_NOMINAL - h)

        d_min = 1e6
        # Walls (always full height).
        if np.isfinite(c.wall_y_min):
            d_min = min(d_min, y - c.wall_y_min - a_eff, c.wall_y_max - y - a_eff)

        arms = [
            self._arm_pos(x, y, heading, b.a_left, 1.0),
            self._arm_pos(x, y, heading, b.a_right, -1.0),
        ]
        if np.isfinite(c.wall_y_min):
            for ax, ay in arms:
                d_min = min(d_min, ay - c.wall_y_min - ARM_RADIUS, c.wall_y_max - ay - ARM_RADIUS)

        for o in c.obstacles:
            z_w = _z_overlap(h, o.z_min, o.z_max)
            if z_w < 1e-4:
                continue
            # Torso ellipse.
            ox = o.cx if o.shape in ("sphere", "qc") else 0.5 * (o.x_min + o.x_max)
            oy = o.cy if o.shape in ("sphere", "qc") else 0.5 * (o.y_min + o.y_max)
            delta_phi = float(np.arctan2(oy - y, ox - x)) - heading
            r_eff = _ellipse_radius(delta_phi, a_eff, TORSO_B)
            d_torso = (_obs_point_sdf(x, y, o) - r_eff) / max(z_w, 1e-6)
            d_min = min(d_min, d_torso)
            # Arm tips.
            for ax, ay in arms:
                d_arm = (_obs_point_sdf(ax, ay, o) - ARM_RADIUS) / max(z_w, 1e-6)
                d_min = min(d_min, d_arm)

        return float(d_min)

    def _sdf_grad_xy(self, x: float, y: float, yaw: float, c: BodySdfContext) -> np.ndarray:
        """Central-difference spatial gradient d(SDF)/d(x, y)."""
        e = self.grad_eps
        gx = (self.body_min_sdf(x + e, y, yaw, c) - self.body_min_sdf(x - e, y, yaw, c)) / (2 * e)
        gy = (self.body_min_sdf(x, y + e, yaw, c) - self.body_min_sdf(x, y - e, yaw, c)) / (2 * e)
        return np.array([gx, gy], dtype=float)

    # -- AdmissibleSet --------------------------------------------------------
    def reference_frame(self, ctx: Any) -> Frame:
        # World frame: (s, n, psi) == (x, y, yaw).
        return Frame(origin=np.zeros(2), yaw=0.0)

    def safety_box(self, r_ref_frame: np.ndarray, ctx: Any, cfg: Any) -> Box:
        """Cheap always-on corridor-edge box (centre clearance only).

        This reproduces the lateral wall guard; the obstacle geometry is handled
        by :meth:`safety_halfspace`. Inactive when walls are unknown.
        """
        c = self._resolve_ctx(ctx)
        if not np.isfinite(c.wall_y_min):
            return Box.inactive()
        a_eff = TORSO_A + TORSO_CROUCH_EXTRA * max(0.0, H_NOMINAL - c.body.h)
        n_lo = c.wall_y_min + a_eff + float(cfg.m_track)
        n_hi = c.wall_y_max - a_eff - float(cfg.m_track)
        if n_lo > n_hi:  # corridor narrower than the body: degenerate, keep centre
            mid = 0.5 * (c.wall_y_min + c.wall_y_max)
            n_lo = n_hi = mid
        return Box(lo=np.array([-np.inf, n_lo, -np.inf]), hi=np.array([np.inf, n_hi, np.inf]))

    # -- lateral set-point search (max-clearance lane) ------------------------
    def _best_lateral_offset(self, x_pass, yaw, c, y_lo, y_hi, y_anchor, n=121):
        """Lateral offset in ``[y_lo, y_hi]`` maximising body-SDF at ``x_pass``.

        Returns ``(y_best, sdf_best)``. Ties (flat plateaus / multiple equal
        maxima, e.g. a symmetric channel) are broken toward ``y_anchor`` (the
        current lateral offset) so the set-point stays continuous and reversible
        across steps -- no sticky side commit, so the reference can weave between
        staggered obstacles (zone_d) without dithering.
        """
        if y_hi <= y_lo + 1e-6:
            yc = 0.5 * (y_lo + y_hi)
            return yc, self.body_min_sdf(x_pass, yc, yaw, c)
        ys = np.linspace(y_lo, y_hi, int(n))
        best_sdf = -1e18
        best_y = y_anchor
        for y in ys:
            d = self.body_min_sdf(x_pass, float(y), yaw, c)
            # Strictly-better updates the max; near-ties prefer the y closest to
            # the anchor (keeps the lane stable and lets it reverse smoothly).
            if d > best_sdf + 1e-4 or (
                abs(d - best_sdf) <= 1e-4 and abs(y - y_anchor) < abs(best_y - y_anchor)
            ):
                best_sdf, best_y = d, float(y)
        return best_y, best_sdf

    def safety_halfspace(self, r_ref_frame: np.ndarray, ctx: Any, cfg: Any) -> Halfspace:
        """Linearised body-SDF safety half-space ``a . r >= b`` (world frame).

        Steers the base reference toward the *globally-consistent safe lane* in a
        short look-ahead window, but only when the executed body would otherwise
        penetrate / graze an obstacle. Construction:

        1. Find the most-constraining longitudinal slice in ``[~now, lookahead]``
           (smallest body-SDF at the current lateral offset).
        2. Target lateral offset at that slice:
           - if the planner trajectory (a collision-free lane) is known, use its
             lateral offset ``y_plan(x_slice)`` -- the plan already picked a
             feasible pass side, which avoids the myopic-gradient failure of
             flipping sides near an obstacle;
           - then refine toward higher clearance with a *one-sided* local search
             (only on the plan's side of centre) so the reference gains margin
             against tracking lag without crossing to the wrong side.
           - if no plan lane, fall back to the local max-clearance search.
        3. Stay inactive when the current offset already gives enough clearance
           and is on the safe side of the target (feedforward preserved; centred
           channels such as zone_c stay centred).
        4. Cap the set-point to a RATE-FEASIBLE lateral step so the projection
           does not demand an infeasible swing and overshoot.
        """
        c = self._resolve_ctx(ctx)
        x0, y0 = float(r_ref_frame[0]), float(r_ref_frame[1])
        yaw0 = float(r_ref_frame[2])
        m = float(cfg.m_track)

        # --- near-passthrough backstop (overrides the legacy lane-steering) ---
        # Inactive unless the body is within m_track + activation_band of an
        # obstacle; then impose only the MINIMAL linearised constraint
        #     SDF(r0) + grad_xy . (r - r0) >= m_track
        # (push along the spatial SDF gradient, away from the nearest obstacle,
        # just enough to restore m_track). No max-clearance lane steering, so
        # with comfortable clearance the governor is a pure passthrough and
        # tracking is preserved. A short look-ahead starts the nudge a step early.
        _sdf0 = self.body_min_sdf(x0, y0, yaw0, c)
        _xe, _sw = x0, _sdf0
        for _i in range(1, 7):
            _xa = x0 + (_i / 6.0) * float(self.lookahead)
            _d = self.body_min_sdf(_xa, y0, yaw0, c)
            if _d < _sw:
                _sw, _xe = _d, _xa
        self.last_sdf = _sdf0
        self.last_grad = None
        if _sw >= m + self.activation_band:
            return Halfspace.inactive()
        _g = self._sdf_grad_xy(_xe, y0, yaw0, c)
        _gn = float(np.hypot(_g[0], _g[1]))
        if _gn < 1e-6:
            return Halfspace.inactive()
        _gy = float(_g[1])
        if abs(_gy) < 1e-6:
            return Halfspace.inactive()
        _side = 1.0 if _gy > 0 else -1.0
        # Lateral-only push (world-y). Forcing motion along the full 2D SDF
        # gradient couples a backward (-x) component when the obstacle is ahead,
        # which stalls the walk (endpoint blows up) and keeps the body grazing
        # the obstacle; restrict the correction to the corridor's free lateral
        # axis so it never fights forward progress.  Required lateral move to
        # restore m_track: side*(y - y0) >= (m - sdf_w)/|gy|.
        _a = np.array([0.0, _side, 0.0], dtype=float)
        self.last_grad = _g
        return Halfspace(a=_a, b=float(_side * y0 + (m - _sw) / abs(_gy)))

        sdf0 = self.body_min_sdf(x0, y0, yaw0, c)
        self.last_sdf = sdf0
        self.last_grad = None

        # The corridor is world-x aligned (obstacles arranged along +x, walls at
        # constant world-y). The safety constraint is therefore a WORLD-Y
        # half-space, decoupled from the robot heading -- using the robot-frame
        # lateral (``[-sin,cos]``) couples the push into the longitudinal axis
        # when the body is yawed and the box-halfspace projection then trades
        # forward progress for lateral safety (the robot stalls / walks back).
        lat = np.array([0.0, 1.0], dtype=float)  # +world-y

        # Lateral band the pelvis CENTRE may occupy (room for the torso ellipse
        # half-width so the target lane is reachable).
        a_eff = TORSO_A + TORSO_CROUCH_EXTRA * max(0.0, H_NOMINAL - c.body.h)
        if np.isfinite(c.wall_y_min):
            y_lo = c.wall_y_min + a_eff
            y_hi = c.wall_y_max - a_eff
            cen = 0.5 * (c.wall_y_min + c.wall_y_max)
        else:
            y_lo, y_hi, cen = -1e3, 1e3, 0.0
        if y_lo > y_hi:
            y_lo = y_hi = cen
        y_lat0 = y0  # current world-y offset

        # --- most-constraining longitudinal slice in the look-ahead window ---
        # Sample along world-x (the corridor direction) at the reference's
        # world-y; keep the slice with the smallest body-SDF.
        worst_slice = None  # (sdf, x_slice, y_slice)
        n_look = 9
        for i in range(n_look + 1):
            xa = x0 + (i / n_look) * float(self.lookahead) - 0.10  # start behind
            ya = y0
            d = self.body_min_sdf(xa, ya, yaw0, c)
            if worst_slice is None or d < worst_slice[0]:
                worst_slice = (d, xa, ya)

        if worst_slice is None or worst_slice[0] >= m + self.activation_band:
            return Halfspace.inactive()  # body clear through the horizon

        sdf_slice, x_pass, y_pass = worst_slice

        # --- target lateral offset at the constraining slice -----------------
        # Track the planner's (collision-free) lane y_plan(x_pass) and push a
        # little FURTHER in the plan's own chosen direction to pre-compensate the
        # policy's lateral tracking lag (the baseline penetrates only because the
        # executed pelvis lags inboard of the plan lane). Crucially the push is
        # in the plan's direction-from-centre only -- we never flip to hunt a
        # different, off-plan lane (that destabilised zone_c, a centred channel
        # where the plan stays put and uses posture, not lateral motion). The
        # overshoot is then refined toward higher clearance within a tight band.
        y_plan = self._plan_lane_y(c, x_pass)
        refine = 0.10
        if y_plan is not None:
            y_plan = float(np.clip(y_plan, y_lo, y_hi))
            dev = y_plan - cen
            dirn = 0.0 if abs(dev) < 1e-3 else (1.0 if dev > 0 else -1.0)
            # Small anticipatory overshoot in the plan's own direction to
            # pre-compensate lateral tracking lag (centred plans -> no push).
            y_over = y_plan + dirn * min(self.lag_margin, abs(dev))
            y_over = float(np.clip(y_over, y_lo, y_hi))
            # Refine toward the highest-clearance offset in a tight band that
            # straddles BOTH the plan lane and the overshoot. Searching both
            # sides lets a centred plan nudge to whichever side actually clears
            # (zone_c: the arm pokes one wall, so the safe nudge is away from it)
            # while the band stays tight enough never to hunt a different lane.
            lo_b = min(y_plan, y_over)
            hi_b = max(y_plan, y_over)
            seg_lo = max(y_lo, lo_b - refine)
            seg_hi = min(y_hi, hi_b + refine)
            y_tgt, _ = self._best_lateral_offset(x_pass, yaw0, c, seg_lo, seg_hi, y_over)
        else:
            seg_lo = max(y_lo, y_lat0 - 2.0 * refine)
            seg_hi = min(y_hi, y_lat0 + 2.0 * refine)
            y_tgt, _ = self._best_lateral_offset(x_pass, yaw0, c, seg_lo, seg_hi, y_lat0)

        y_lat_tgt = y_tgt  # world-y target
        side = 1.0 if y_lat_tgt >= y_lat0 else -1.0

        # Already on the safe side of the target with adequate clearance: skip.
        if side * (y_lat_tgt - y_lat0) <= 1e-3:
            return Halfspace.inactive()

        # Cap to a RATE-FEASIBLE lateral step (the policy tracks with finite
        # lateral speed; demanding more just saturates the box and overshoots).
        lon_to_slice = max(0.05, float(x_pass - x0))
        v_lon = max(0.1, float(cfg.v_max_lon))
        t_to_slice = lon_to_slice / v_lon
        reach = float(cfg.v_max_lat) * (t_to_slice + 2.0 * float(cfg.dt))
        y_lat_target = y_lat0 + side * min(abs(y_lat_tgt - y_lat0), reach)

        a_vec = side * lat
        a = np.array([a_vec[0], a_vec[1], 0.0], dtype=float)
        b = side * y_lat_target
        self.last_grad = np.array([a[0], a[1]], dtype=float)
        return Halfspace(a=a, b=b)
