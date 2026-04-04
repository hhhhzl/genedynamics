"""
Humanoid narrow-corridor obstacle avoidance 2D planning environment.

State (14D):
  [x, y, psi, h, psi_torso,
   a_L, a_R, p_L, p_R,
   v_x, v_y, omega, h_dot, psi_dot_torso]

  x, y         : base position in world frame (m)
  psi          : base yaw (rad)
  h            : body height / crouch (m), nominal ~0.75
  psi_torso    : torso yaw relative to base (rad)
  a_L, a_R     : left/right arm tuck [0=open, 1=tucked]
  p_L, p_R     : left/right arm posture latent [-1, 1]
  v_x, v_y     : base velocity (m/s) in world frame
  omega        : yaw rate (rad/s)
  h_dot        : height rate (m/s)
  psi_dot_torso: torso yaw rate (rad/s)

Action (9D):
  [v_x_cmd, v_y_cmd, omega_cmd, h_dot_cmd, psi_dot_torso_cmd,
   a_dot_L, a_dot_R, p_dot_L, p_dot_R]

Dynamics: velocity-rate integrator (control = desired velocities/rates).

Collision model: elliptical torso + two arm tips, all height-gated
against 3D bounding-box obstacles.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except Exception:
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]

from genedynamics.core.energy import EnergyTerm, LegacyEnergyFunctional

# ---------------------------------------------------------------------------
# State / action layout
# ---------------------------------------------------------------------------
_S_X = 0
_S_Y = 1
_S_PSI = 2
_S_H = 3
_S_PSI_T = 4
_S_AL = 5
_S_AR = 6
_S_PL = 7
_S_PR = 8
_S_VX = 9
_S_VY = 10
_S_OM = 11
_S_HD = 12
_S_PTD = 13
STATE_DIM = 14

_A_VX = 0
_A_VY = 1
_A_OM = 2
_A_HD = 3
_A_PTD = 4
_A_ADL = 5
_A_ADR = 6
_A_PDL = 7
_A_PDR = 8
ACT_DIM = 9

# ---------------------------------------------------------------------------
# G1 body geometry constants
# ---------------------------------------------------------------------------
# Torso ellipse semi-axes (half shoulder-width, half chest-depth).
TORSO_A = 0.20  # front half-width (m)
TORSO_B = 0.11  # side half-depth (m)
# Additional torso radius growth when crouching (torso leans forward).
TORSO_CROUCH_EXTRA = 0.03
# Arm reach as a function of tuck parameter a ∈ [0, 1].
ARM_REACH_OPEN = 0.35   # a = 0
ARM_REACH_TUCKED = 0.10  # a = 1
ARM_RADIUS = 0.05  # collision sphere radius at arm tip
# Body half-height for z-overlap computation.
BODY_HALF_H = 0.25
# Nominal body height.
H_NOMINAL = 0.75
H_MIN = 0.55
H_MAX = 0.85

# Effective width model for squeeze feasibility.
# W_eff(z) = W0 - alpha_torso * |psi_torso| - alpha_arm * (a_L + a_R)
# W0: torso frontal width (shoulder width = 2*TORSO_A = 0.40m).
W_EFF_W0 = 2 * TORSO_A  # 0.40m facing forward
# At 90 deg torso yaw: effective width drops to 2*TORSO_B = 0.22m.
# Delta = 0.40 - 0.22 = 0.18m over 1.57 rad.
W_EFF_ALPHA_TORSO = (2 * TORSO_A - 2 * TORSO_B) / 1.57  # ~0.115 per radian
# Arm tuck reduces lateral protrusion. Each arm at a=0 extends ~0.05m beyond shoulder.
# At a=1 it's flush. So each arm saves ~0.05m.
W_EFF_ALPHA_ARM = 0.05  # per unit a (each arm)


# ---------------------------------------------------------------------------
# Obstacle specification
# ---------------------------------------------------------------------------
@dataclass
class CorridorObstacle:
    """Axis-aligned 3D bounding-box obstacle in the corridor."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float = 0.0
    z_max: float = 2.0
    name: str = ""
    shape: str = "box"  # "box", "sphere", or "qc"
    # Sphere / quarter-circle fields (cx, cy, radius shared by both).
    cx: float = 0.0
    cy: float = 0.0
    radius: float = 0.0
    # Quarter-circle clip direction: +1 = entry (solid at x<=cx), -1 = exit (solid at x>=cx).
    qc_clip_sign: float = 0.0

    @classmethod
    def sphere(cls, cx: float, cy: float, radius: float,
               z_min: float = 0.0, z_max: float = 2.0,
               name: str = "") -> "CorridorObstacle":
        """Convenience constructor for a spherical obstacle.

        Internally stores a tight bounding box so that existing AABB code
        can quickly skip distant obstacles, while the SDF uses the true
        circle distance.
        """
        return cls(
            x_min=cx - radius, x_max=cx + radius,
            y_min=cy - radius, y_max=cy + radius,
            z_min=z_min, z_max=z_max,
            name=name, shape="sphere",
            cx=cx, cy=cy, radius=radius,
        )

    @classmethod
    def quarter_circle(cls, cx: float, cy: float, radius: float,
                       clip_sign: float,
                       z_min: float = 0.0, z_max: float = 2.0,
                       name: str = "") -> "CorridorObstacle":
        """Quarter-circle obstacle at a wall-block corner.

        SDF = max(circle_sdf, clip_sign * (px - cx)).
        clip_sign = +1: solid region at x <= cx (entry taper).
        clip_sign = -1: solid region at x >= cx (exit taper).
        """
        x_min = cx - radius if clip_sign > 0 else cx
        x_max = cx if clip_sign > 0 else cx + radius
        y_min = cy - radius if cy > 0 else cy
        y_max = cy if cy > 0 else cy + radius
        return cls(
            x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max,
            z_min=z_min, z_max=z_max,
            name=name, shape="qc",
            cx=cx, cy=cy, radius=radius,
            qc_clip_sign=float(clip_sign),
        )


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
@dataclass
class CorridorScene:
    """Complete corridor scene with walls and obstacles."""
    corridor_width: float = 1.6
    corridor_length: float = 10.0
    start_pos: Tuple[float, float] = (0.5, 0.0)
    goal_pos: Tuple[float, float] = (9.5, 0.0)
    obstacles: List[CorridorObstacle] = field(default_factory=list)

    @property
    def wall_y_min(self) -> float:
        return -self.corridor_width / 2.0

    @property
    def wall_y_max(self) -> float:
        return self.corridor_width / 2.0

    # Compatibility attributes for generic visualization plugins.
    @property
    def map_x(self) -> Tuple[float, float]:
        return (0.0, self.corridor_length)

    @property
    def map_y(self) -> Tuple[float, float]:
        return (self.wall_y_min, self.wall_y_max)

    # ------------------------------------------------------------------
    # Pre-built difficulty presets
    # ------------------------------------------------------------------
    @classmethod
    def easy(cls) -> "CorridorScene":
        """Wide corridor, 2 side protrusions only."""
        return cls(
            corridor_width=1.6,
            corridor_length=8.0,
            start_pos=(0.5, 0.0),
            goal_pos=(7.5, 0.0),
            obstacles=[
                CorridorObstacle(1.5, 2.5, 0.15, 0.60, 0.0, 2.0, "protrusion_L"),
                CorridorObstacle(3.0, 4.0, -0.80, -0.20, 0.0, 2.0, "protrusion_R"),
            ],
        )

    @classmethod
    def medium(cls) -> "CorridorScene":
        """4-zone layout: thin wall, U-wall, squeeze with qc ends, aerial spheres."""
        hw = 0.80   # corridor half-width
        squeeze_inner = 0.35  # half-gap at narrowest → gap = 0.70m
        # Quarter-circle radius = wall-to-squeeze distance.
        qr = hw - squeeze_inner  # 0.45m
        # Zone B U-wall: hangs from top wall, blocks 60% of corridor.
        # Robot must detour toward bottom wall (U-shape path).
        u_wall_bottom = -0.16  # extends from top wall (0.80) to -0.16 → blocks 0.96m (60%)
        return cls(
            corridor_width=1.6,
            corridor_length=8.0,
            start_pos=(0.5, 0.0),
            goal_pos=(7.5, 0.0),
            obstacles=[
                # Zone A: thin wall (centred, both sides passable)
                CorridorObstacle(1.9, 2.2, -0.05, 0.05, 0.0, 2.0, "lateral_wall"),
                # Zone B: U-wall connected to top wall, gap at bottom (0.64m)
                # CorridorObstacle(2.8, 2.95, u_wall_bottom, hw, 0.0, 2.0, "u_wall"),
                # Zone C: 1m block + 1/4-circle entry/exit per side.
                CorridorObstacle(4, 5, squeeze_inner, hw, 0.0, 2.0, "squeeze_L"),
                CorridorObstacle(4, 5, -hw, -squeeze_inner, 0.0, 2.0, "squeeze_R"),
                CorridorObstacle.quarter_circle(4.0, hw, qr, +1.0, 0.0, 2.0, "qc_L_entry"),
                CorridorObstacle.quarter_circle(4.0, -hw, qr, +1.0, 0.0, 2.0, "qc_R_entry"),
                CorridorObstacle.quarter_circle(5.0, hw, qr, -1.0, 0.0, 2.0, "qc_L_exit"),
                CorridorObstacle.quarter_circle(5.0, -hw, qr, -1.0, 0.0, 2.0, "qc_R_exit"),
                # Zone D: 2 small aerial spheres (arm-sized, laterally offset)
                CorridorObstacle.sphere(6.0, 0.3, 0.05, 0.40, 1.05, "ball_L1"),
                CorridorObstacle.sphere(7.0, -0.3, 0.05, 0.45, 1.00, "ball_R1"),
            ],
        )

    @classmethod
    def zone_abc(cls) -> "CorridorScene":
        """Zones A+B+C combined: thin wall, high bar, wide squeeze. 6m corridor."""
        hw = 0.80
        return cls(
            corridor_width=1.6, corridor_length=6.0,
            start_pos=(0.5, 0.0), goal_pos=(5.5, 0.0),
            obstacles=[
                # Zone A: thin center wall at x=1.5
                CorridorObstacle(1.5, 1.6, -0.15, 0.15, 0.0, 2.0, "thin_wall"),
                # Zone B: high bar at x=2.8 (z_min=0.90)
                CorridorObstacle(2.8, 3.1, -hw, hw, 0.90, 2.0, "low_bar"),
                # Zone C: squeeze at x=3.8-4.8 (gap=0.60m)
                CorridorObstacle(3.8, 4.8, 0.30, hw, 0.0, 2.0, "squeeze_L"),
                CorridorObstacle(3.8, 4.8, -hw, -0.30, 0.0, 2.0, "squeeze_R"),
            ],
        )

    # ------------------------------------------------------------------
    # Single-zone presets for isolated testing
    # ------------------------------------------------------------------
    @classmethod
    def zone_a(cls) -> "CorridorScene":
        """Zone A only: lateral wall (0.3m x 0.1m), wide corridor for easy passage both sides."""
        return cls(
            corridor_width=1.6, corridor_length=4.0,
            start_pos=(0.5, 0.0), goal_pos=(3.5, 0.0),
            obstacles=[
                # 0.3m(x) × 0.1m(y), centered. Each passage 0.75m wide — no sidestep needed.
                CorridorObstacle(1.7, 2.0, -0.05, 0.05, 0.0, 2.0, "lateral_wall"),
            ],
        )

    @classmethod
    def zone_b(cls) -> "CorridorScene":
        """Zone B only: U-wall from top wall, blocks 60%, gap at bottom. 4m corridor."""
        hw = 0.80
        u_wall_bottom = -0.16  # blocks 0.96m (60%), gap = 0.64m at bottom
        return cls(
            corridor_width=1.6, corridor_length=4.0,
            start_pos=(1, 0.1), goal_pos=(3, 0.25),
            obstacles=[
                CorridorObstacle(1.8, 1.95, u_wall_bottom, hw, 0.0, 2.0, "u_wall"),
            ],
        )

    @classmethod
    def zone_c(cls) -> "CorridorScene":
        """Zone C only: taper → 1m squeeze → taper. Gap=0.60m at narrowest."""
        hw = 0.80
        squeeze_inner = 0.30  # ±0.30 from center → 0.60m gap
        obs = [
            # Main squeeze section (1.0m long in x)
            CorridorObstacle(1.5, 2.5, squeeze_inner, hw, 0.0, 2.0, "squeeze_L"),
            CorridorObstacle(1.5, 2.5, -hw, -squeeze_inner, 0.0, 2.0, "squeeze_R"),
            # Entry taper (2 slices per side)
            CorridorObstacle(1.1, 1.3, 0.55, hw, 0.0, 2.0, "taper_L_entry1"),
            CorridorObstacle(1.3, 1.5, squeeze_inner + 0.12, hw, 0.0, 2.0, "taper_L_entry2"),
            CorridorObstacle(1.1, 1.3, -hw, -0.55, 0.0, 2.0, "taper_R_entry1"),
            CorridorObstacle(1.3, 1.5, -hw, -(squeeze_inner + 0.12), 0.0, 2.0, "taper_R_entry2"),
            # Exit taper (2 slices per side)
            CorridorObstacle(2.5, 2.7, squeeze_inner + 0.12, hw, 0.0, 2.0, "taper_L_exit1"),
            CorridorObstacle(2.7, 2.9, 0.55, hw, 0.0, 2.0, "taper_L_exit2"),
            CorridorObstacle(2.5, 2.7, -hw, -(squeeze_inner + 0.12), 0.0, 2.0, "taper_R_exit1"),
            CorridorObstacle(2.7, 2.9, -hw, -0.55, 0.0, 2.0, "taper_R_exit2"),
        ]
        return cls(
            corridor_width=2 * hw, corridor_length=4.0,
            start_pos=(0.5, 0.0), goal_pos=(3.5, 0.0),
            obstacles=obs,
        )

    @classmethod
    def zone_d(cls) -> "CorridorScene":
        """Zone D only: 2 small aerial spheres, but SDF treated as full-height for planning."""
        return cls(
            corridor_width=1.6, corridor_length=4.0,
            start_pos=(0.5, 0.0), goal_pos=(3.5, 0.0),
            obstacles=[
                # Visual: aerial spheres. SDF: full-height (z=0,2) so CFS projects around them.
                # Displayed as aerial but collision model is full-height cylinder.
                CorridorObstacle.sphere(2.0, 0.25, 0.06, 0.0, 2.0, "ball_L1"),
                CorridorObstacle.sphere(2.5, -0.25, 0.06, 0.0, 2.0, "ball_R1"),
            ],
        )

    @classmethod
    def hard(cls) -> "CorridorScene":
        """Narrower corridor, tighter squeeze, more obstacles."""
        hw = 0.50
        return cls(
            corridor_width=1.0,
            corridor_length=10.0,
            start_pos=(0.5, 0.0),
            goal_pos=(9.5, 0.0),
            obstacles=[
                # Zone A: tighter staggered protrusions
                CorridorObstacle(1.0, 2.0, 0.10, hw, 0.0, 2.0, "protrus_L1"),
                CorridorObstacle(1.5, 2.5, -hw, -0.10, 0.0, 2.0, "protrus_R1"),
                CorridorObstacle(2.5, 3.0, 0.05, hw, 0.0, 2.0, "protrus_L2"),
                # Zone B: low bar (crouch under) + ground block (step around)
                CorridorObstacle(3.2, 3.5, -hw, hw, 0.82, 2.0, "low_bar"),
                CorridorObstacle(3.8, 4.3, -0.10, hw, 0.0, 0.28, "ground_block"),
                # Zone C: very tight trapezoid (gap = 0.24m, must sidestep)
                CorridorObstacle(4.5, 5.0, 0.12, hw, 0.0, 2.0, "trap_L_entry"),
                CorridorObstacle(5.0, 5.5, 0.12, hw, 0.0, 2.0, "trap_L_mid"),
                CorridorObstacle(5.5, 6.0, 0.12, hw, 0.0, 2.0, "trap_L_exit"),
                CorridorObstacle(4.5, 5.0, -hw, -0.12, 0.0, 2.0, "trap_R_entry"),
                CorridorObstacle(5.0, 5.5, -hw, -0.12, 0.0, 2.0, "trap_R_mid"),
                CorridorObstacle(5.5, 6.0, -hw, -0.12, 0.0, 2.0, "trap_R_exit"),
                # Zone D: aerial spheres + ground
                CorridorObstacle.sphere(6.8, 0.15, 0.17, 0.40, 1.05, "ball_L1"),
                CorridorObstacle.sphere(7.3, -0.10, 0.16, 0.42, 0.95, "ball_R1"),
                CorridorObstacle.sphere(7.8, 0.05, 0.18, 0.40, 1.00, "ball_L2"),
                CorridorObstacle.sphere(8.3, -0.20, 0.15, 0.45, 0.98, "ball_R2"),
                CorridorObstacle.sphere(8.7, 0.15, 0.13, 0.40, 0.95, "ball_L3"),
                CorridorObstacle(7.0, 7.5, -hw, -0.20, 0.0, 2.0, "squeeze_R"),
                CorridorObstacle(8.5, 9.0, 0.20, hw, 0.0, 0.45, "floor_block2"),
            ],
        )

    @classmethod
    def trapezoid_squeeze(cls) -> "CorridorScene":
        """Tapered walls that linearly converge then diverge.

        Instead of axis-aligned boxes we approximate the taper with a series
        of thin slices so that the SDF stays box-based and JAX-friendly.
        Each slice spans 0.25 m in x; the inner y-boundary linearly
        interpolates from the corridor wall to the narrowest gap.
        """
        hw = 0.80
        gap_half = 0.14          # half-gap at the narrowest point
        taper_x_start = 4.5
        taper_x_end = 6.0
        taper_x_mid = 0.5 * (taper_x_start + taper_x_end)
        n_slices = 12
        dx = (taper_x_end - taper_x_start) / n_slices
        obs: List[CorridorObstacle] = []
        for i in range(n_slices):
            xl = taper_x_start + i * dx
            xr = xl + dx
            xc = 0.5 * (xl + xr)
            # Linear taper: wall → gap_half → wall
            t = abs(xc - taper_x_mid) / (0.5 * (taper_x_end - taper_x_start))
            inner_y = gap_half + t * (hw - gap_half)
            obs.append(CorridorObstacle(xl, xr, inner_y, hw, 0.0, 2.0, f"taper_L_{i}"))
            obs.append(CorridorObstacle(xl, xr, -hw, -inner_y, 0.0, 2.0, f"taper_R_{i}"))
        return cls(
            corridor_width=2 * hw,
            corridor_length=10.0,
            start_pos=(0.5, 0.0),
            goal_pos=(9.5, 0.0),
            obstacles=obs,
        )


def resolve_corridor_scene_preset(scene_preset: str) -> CorridorScene:
    """
    Resolve a named corridor preset into a concrete scene object.

    This is used by planning, replay, and offline rendering code so that
    visualizations can reconstruct the same corridor geometry as the planner.
    """
    preset = str(scene_preset or "").strip().lower()
    table = {
        "easy": CorridorScene.easy,
        "medium": CorridorScene.medium,
        "hard": CorridorScene.hard,
        "zone_a": CorridorScene.zone_a,
        "zone_b": CorridorScene.zone_b,
        "zone_c": CorridorScene.zone_c,
        "zone_d": CorridorScene.zone_d,
        "zone_abc": CorridorScene.zone_abc,
    }
    if preset not in table:
        raise KeyError(f"Unknown corridor scene preset: {scene_preset}")
    return table[preset]()


def corridor_scene_to_dict(scene: CorridorScene, *, scene_preset: Optional[str] = None) -> Dict[str, Any]:
    """
    Serialize a corridor scene into a JSON-friendly dictionary.
    """
    blob = asdict(scene)
    if scene_preset is not None:
        blob["scene_preset"] = str(scene_preset)
    return blob


# ---------------------------------------------------------------------------
# Numpy helpers
# ---------------------------------------------------------------------------
def _to_np(x: Any, n: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).ravel()
    if arr.size < n:
        arr = np.pad(arr, (0, n - arr.size))
    return arr[:n]


def _wrap(theta: float) -> float:
    return float(np.arctan2(np.sin(theta), np.cos(theta)))


def _ellipse_radius(delta_phi: float, a: float = TORSO_A, b: float = TORSO_B) -> float:
    """Effective collision radius of an ellipse at bearing *delta_phi*."""
    cd = float(np.cos(delta_phi))
    sd = float(np.sin(delta_phi))
    return 1.0 / np.sqrt((cd / a) ** 2 + (sd / b) ** 2 + 1e-12)


def _arm_reach(a_tuck: float) -> float:
    return ARM_REACH_OPEN + (ARM_REACH_TUCKED - ARM_REACH_OPEN) * float(np.clip(a_tuck, 0.0, 1.0))


def _z_overlap(h: float, z_lo: float, z_hi: float) -> float:
    """Fraction of body vertical extent [h-R, h+R] overlapping [z_lo, z_hi]."""
    body_lo = h - BODY_HALF_H
    body_hi = h + BODY_HALF_H
    overlap = min(body_hi, z_hi) - max(body_lo, z_lo)
    return float(np.clip(overlap / (2.0 * BODY_HALF_H), 0.0, 1.0))


def _box_sdf_2d(px: float, py: float, xmin: float, xmax: float, ymin: float, ymax: float) -> float:
    """Signed distance from point (px, py) to the exterior of an AABB.
    Positive outside, negative inside."""
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    hx = 0.5 * (xmax - xmin)
    hy = 0.5 * (ymax - ymin)
    dx = abs(px - cx) - hx
    dy = abs(py - cy) - hy
    outside = np.sqrt(max(dx, 0.0) ** 2 + max(dy, 0.0) ** 2 + 1e-12)
    inside = min(max(dx, dy), 0.0)
    return float(outside + inside)


def _sphere_sdf_2d(px: float, py: float, cx: float, cy: float, r: float) -> float:
    """Signed distance from point to exterior of a circle. Positive outside."""
    return float(np.sqrt((px - cx) ** 2 + (py - cy) ** 2 + 1e-12) - r)


def _qc_sdf_2d(px: float, py: float, cx: float, cy: float, r: float, clip_sign: float) -> float:
    """SDF for a quarter-circle obstacle. max(circle, x-clip, y-clip)."""
    circle_d = float(np.sqrt((px - cx) ** 2 + (py - cy) ** 2 + 1e-12) - r)
    x_hp = float(clip_sign * (px - cx))
    y_hp = float(py - cy) if cy > 0 else float(cy - py)
    return max(circle_d, x_hp, y_hp)


def _obs_sdf_2d_np(px: float, py: float, obs: "CorridorObstacle") -> float:
    """Unified SDF for a single obstacle (box, sphere, or quarter-circle)."""
    if obs.shape == "sphere":
        return _sphere_sdf_2d(px, py, obs.cx, obs.cy, obs.radius)
    if obs.shape == "qc":
        return _qc_sdf_2d(px, py, obs.cx, obs.cy, obs.radius, obs.qc_clip_sign)
    return _box_sdf_2d(px, py, obs.x_min, obs.x_max, obs.y_min, obs.y_max)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
@dataclass
class HumanoidCorridor2DEnv:
    """Humanoid corridor obstacle avoidance planning environment (14D state, 9D action)."""

    dt: float = 0.2
    horizon: int = 30

    scene_preset: str = "medium"
    scene: Optional[CorridorScene] = None

    # Control bounds
    vx_max: float = 0.8
    vx_min: float = -0.3
    vy_max: float = 0.3
    omega_max: float = 0.5
    h_dot_max: float = 0.3
    psi_torso_dot_max: float = 0.4
    arm_tuck_rate_max: float = 1.0
    arm_posture_rate_max: float = 1.0

    # State bounds
    psi_torso_max: float = 1.57  # ±90 deg
    psi_max: float = 3.14

    # Collision safety margin
    collision_margin: float = 0.03

    state_dim: int = STATE_DIM
    act_dim: int = ACT_DIM

    # CFS / CBF integration hints
    enable_cfs_safety_points_qp: bool = False
    robot_radius: float = 0.25  # torso half-width + margin for arms
    cfs_qp_num_safety_points: int = 5  # torso, arm_L, arm_R, arm_L_mid, arm_R_mid
    cfs_qp_safety_clearance_margin_only: bool = False

    def __post_init__(self) -> None:
        if self.scene is None:
            presets = {
                "easy": CorridorScene.easy,
                "medium": CorridorScene.medium,
                "hard": CorridorScene.hard,
                "trapezoid": CorridorScene.trapezoid_squeeze,
                "zone_abc": CorridorScene.zone_abc,
                "zone_a": CorridorScene.zone_a,
                "zone_b": CorridorScene.zone_b,
                "zone_c": CorridorScene.zone_c,
                "zone_d": CorridorScene.zone_d,
            }
            factory = presets.get(self.scene_preset, CorridorScene.medium)
            self.scene = factory()
        self.target = np.asarray(self.scene.goal_pos, dtype=np.float32)
        self.start = np.asarray(self.scene.start_pos, dtype=np.float32)

        # Pack obstacles into arrays for fast vectorised SDF.
        obs = self.scene.obstacles
        n = len(obs)
        self._n_obs = n
        self._obs_xmin = np.array([o.x_min for o in obs], dtype=np.float32) if n else np.zeros(0, dtype=np.float32)
        self._obs_xmax = np.array([o.x_max for o in obs], dtype=np.float32) if n else np.zeros(0, dtype=np.float32)
        self._obs_ymin = np.array([o.y_min for o in obs], dtype=np.float32) if n else np.zeros(0, dtype=np.float32)
        self._obs_ymax = np.array([o.y_max for o in obs], dtype=np.float32) if n else np.zeros(0, dtype=np.float32)
        self._obs_zmin = np.array([o.z_min for o in obs], dtype=np.float32) if n else np.zeros(0, dtype=np.float32)
        self._obs_zmax = np.array([o.z_max for o in obs], dtype=np.float32) if n else np.zeros(0, dtype=np.float32)
        # Sphere flag + geometry (cx, cy, radius). For boxes: is_sphere=0, fields unused.
        self._obs_is_sphere = np.array([1.0 if o.shape == "sphere" else 0.0 for o in obs], dtype=np.float32) if n else np.zeros(0, dtype=np.float32)
        self._obs_cx = np.array([o.cx for o in obs], dtype=np.float32) if n else np.zeros(0, dtype=np.float32)
        self._obs_cy = np.array([o.cy for o in obs], dtype=np.float32) if n else np.zeros(0, dtype=np.float32)
        self._obs_radius = np.array([o.radius for o in obs], dtype=np.float32) if n else np.zeros(0, dtype=np.float32)
        # Quarter-circle flag + clip direction.
        self._obs_is_qc = np.array([1.0 if o.shape == "qc" else 0.0 for o in obs], dtype=np.float32) if n else np.zeros(0, dtype=np.float32)
        self._obs_qc_clip_sign = np.array([o.qc_clip_sign for o in obs], dtype=np.float32) if n else np.zeros(0, dtype=np.float32)

        # Default initial state
        self._default_state = np.zeros(STATE_DIM, dtype=np.float32)
        self._default_state[_S_X] = self.start[0]
        self._default_state[_S_Y] = self.start[1]
        self._default_state[_S_H] = H_NOMINAL

        # Control limits packed for clipping
        self._u_lo = np.array([
            self.vx_min, -self.vy_max, -self.omega_max,
            -self.h_dot_max, -self.psi_torso_dot_max,
            -self.arm_tuck_rate_max, -self.arm_tuck_rate_max,
            -self.arm_posture_rate_max, -self.arm_posture_rate_max,
        ], dtype=np.float32)
        self._u_hi = np.array([
            self.vx_max, self.vy_max, self.omega_max,
            self.h_dot_max, self.psi_torso_dot_max,
            self.arm_tuck_rate_max, self.arm_tuck_rate_max,
            self.arm_posture_rate_max, self.arm_posture_rate_max,
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Collision body helpers (NumPy)
    # ------------------------------------------------------------------
    def _torso_pos_np(self, x: np.ndarray) -> np.ndarray:
        return np.array([x[_S_X], x[_S_Y]], dtype=np.float32)

    def _arm_pos_np(self, x: np.ndarray, side: str) -> np.ndarray:
        """World position of arm tip."""
        a_idx = _S_AL if side == "L" else _S_AR
        a_tuck = float(x[a_idx])
        reach = _arm_reach(a_tuck)
        heading = float(x[_S_PSI]) + float(x[_S_PSI_T])
        sign = 1.0 if side == "L" else -1.0
        # Arm extends laterally in torso frame.
        local = np.array([0.0, sign * reach], dtype=np.float32)
        c, s = np.cos(heading), np.sin(heading)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        return self._torso_pos_np(x) + R @ local

    def _collision_bodies_np(self, x: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """Return [(pos_2d, radius), ...] for torso, arm_L, arm_R."""
        h = float(x[_S_H])
        torso_r = TORSO_A + TORSO_CROUCH_EXTRA * max(0.0, H_NOMINAL - h)
        return [
            (self._torso_pos_np(x), torso_r),
            (self._arm_pos_np(x, "L"), ARM_RADIUS),
            (self._arm_pos_np(x, "R"), ARM_RADIUS),
        ]

    def _obs_point_sdf_np(self, px: float, py: float, obs_idx: int) -> float:
        """SDF from a point to obstacle *obs_idx* (box, sphere, or qc)."""
        if self._obs_is_qc[obs_idx] > 0.5:
            return _qc_sdf_2d(px, py, self._obs_cx[obs_idx], self._obs_cy[obs_idx],
                              self._obs_radius[obs_idx], self._obs_qc_clip_sign[obs_idx])
        if self._obs_is_sphere[obs_idx] > 0.5:
            return _sphere_sdf_2d(px, py, self._obs_cx[obs_idx], self._obs_cy[obs_idx], self._obs_radius[obs_idx])
        return _box_sdf_2d(px, py, self._obs_xmin[obs_idx], self._obs_xmax[obs_idx],
                           self._obs_ymin[obs_idx], self._obs_ymax[obs_idx])

    def _torso_ellipse_sdf_to_obs_np(self, x: np.ndarray, obs_idx: int) -> float:
        """SDF of the torso ellipse to a single obstacle, with height gating."""
        px, py = float(x[_S_X]), float(x[_S_Y])
        h = float(x[_S_H])
        heading = float(x[_S_PSI]) + float(x[_S_PSI_T])
        ox = 0.5 * (self._obs_xmin[obs_idx] + self._obs_xmax[obs_idx])
        oy = 0.5 * (self._obs_ymin[obs_idx] + self._obs_ymax[obs_idx])
        delta_phi = float(np.arctan2(oy - py, ox - px)) - heading
        a_eff = TORSO_A + TORSO_CROUCH_EXTRA * max(0.0, H_NOMINAL - h)
        r_eff = _ellipse_radius(delta_phi, a_eff, TORSO_B)
        d_obs = self._obs_point_sdf_np(px, py, obs_idx)
        z_w = _z_overlap(h, self._obs_zmin[obs_idx], self._obs_zmax[obs_idx])
        if z_w < 1e-4:
            return 1e6
        return (d_obs - r_eff) / max(z_w, 1e-6)

    def _body_min_sdf_np(self, x: np.ndarray) -> float:
        """Minimum SDF across all collision bodies and all obstacles + walls."""
        min_d = 1e6
        h = float(x[_S_H])
        heading = float(x[_S_PSI]) + float(x[_S_PSI_T])
        px, py = float(x[_S_X]), float(x[_S_Y])

        # Wall SDFs (always full height, always active).
        w_lo = self.scene.wall_y_min
        w_hi = self.scene.wall_y_max
        # Torso to walls (use max torso radius for conservative check).
        a_eff = TORSO_A + TORSO_CROUCH_EXTRA * max(0.0, H_NOMINAL - h)
        min_d = min(min_d, py - w_lo - a_eff, w_hi - py - a_eff)
        # Arms to walls
        for side in ("L", "R"):
            ap = self._arm_pos_np(x, side)
            min_d = min(min_d, float(ap[1]) - w_lo - ARM_RADIUS, w_hi - float(ap[1]) - ARM_RADIUS)

        # Obstacle SDFs
        for i in range(self._n_obs):
            # Torso (ellipse)
            d_torso = self._torso_ellipse_sdf_to_obs_np(x, i)
            min_d = min(min_d, d_torso)
            # Arms
            z_w_arm_h = _z_overlap(h, self._obs_zmin[i], self._obs_zmax[i])
            if z_w_arm_h < 1e-4:
                continue
            for side in ("L", "R"):
                ap = self._arm_pos_np(x, side)
                d_arm = self._obs_point_sdf_np(float(ap[0]), float(ap[1]), i)
                min_d = min(min_d, (d_arm - ARM_RADIUS) / max(z_w_arm_h, 1e-6))

        return float(min_d)

    # ------------------------------------------------------------------
    # reset / transition / step  (NumPy)
    # ------------------------------------------------------------------
    def reset(self, rng: Optional[Any] = None, **kwargs) -> Tuple[np.ndarray, dict]:
        return self._default_state.copy(), {}

    def transition(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        x = _to_np(state, STATE_DIM)
        u = np.clip(_to_np(action, ACT_DIM), self._u_lo, self._u_hi)
        dt = self.dt

        psi = float(x[_S_PSI])
        c, s = np.cos(psi), np.sin(psi)

        vx_cmd, vy_cmd = float(u[_A_VX]), float(u[_A_VY])
        nx = float(x[_S_X]) + dt * (vx_cmd * c - vy_cmd * s)
        ny = float(x[_S_Y]) + dt * (vx_cmd * s + vy_cmd * c)
        npsi = _wrap(psi + dt * float(u[_A_OM]))
        nh = float(np.clip(x[_S_H] + dt * u[_A_HD], H_MIN, H_MAX))
        npt = float(np.clip(x[_S_PSI_T] + dt * u[_A_PTD], -self.psi_torso_max, self.psi_torso_max))
        nal = float(np.clip(x[_S_AL] + dt * u[_A_ADL], 0.0, 1.0))
        nar = float(np.clip(x[_S_AR] + dt * u[_A_ADR], 0.0, 1.0))
        npl = float(np.clip(x[_S_PL] + dt * u[_A_PDL], -1.0, 1.0))
        npr = float(np.clip(x[_S_PR] + dt * u[_A_PDR], -1.0, 1.0))

        out = np.zeros(STATE_DIM, dtype=np.float32)
        out[_S_X] = nx
        out[_S_Y] = ny
        out[_S_PSI] = npsi
        out[_S_H] = nh
        out[_S_PSI_T] = npt
        out[_S_AL] = nal
        out[_S_AR] = nar
        out[_S_PL] = npl
        out[_S_PR] = npr
        out[_S_VX] = vx_cmd
        out[_S_VY] = vy_cmd
        out[_S_OM] = float(u[_A_OM])
        out[_S_HD] = float(u[_A_HD])
        out[_S_PTD] = float(u[_A_PTD])
        return out

    def step(self, state: np.ndarray, action: np.ndarray,
             t: Optional[int] = None, info: Optional[dict] = None):
        ns = self.transition(state, action)
        c = self.cost(ns)
        done = bool(np.linalg.norm(ns[:2] - self.target) < 0.15)
        return ns, c, done, {}

    def rollout_actions(self, state: np.ndarray, actions: np.ndarray) -> np.ndarray:
        x = _to_np(state, STATE_DIM)
        traj = [x.copy()]
        for a in np.asarray(actions, dtype=np.float32):
            x = self.transition(x, a)
            traj.append(x.copy())
        return np.asarray(traj, dtype=np.float32)

    def model_transition(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return self.transition(state, action)

    # ------------------------------------------------------------------
    # cost  (NumPy)
    # ------------------------------------------------------------------
    def cost(self, state: np.ndarray) -> float:
        x = _to_np(state, STATE_DIM)
        goal_err = float(np.sum((x[:2] - self.target) ** 2))
        center_err = float(x[_S_Y] ** 2)
        arm_err = float(x[_S_AL] ** 2 + x[_S_AR] ** 2)
        torso_err = float(x[_S_PSI_T] ** 2)
        height_err = float((x[_S_H] - H_NOMINAL) ** 2)
        return goal_err + 2.0 * center_err + arm_err + torso_err + 2.0 * height_err

    def terminal_distance(self, state: np.ndarray) -> float:
        x = _to_np(state, STATE_DIM)
        return float(np.linalg.norm(x[:2] - self.target))

    # ------------------------------------------------------------------
    # Collision checking  (NumPy)
    # ------------------------------------------------------------------
    def check_collision(self, state: np.ndarray) -> bool:
        return self._body_min_sdf_np(_to_np(state, STATE_DIM)) < 0.0

    def get_min_clearance(self, state: np.ndarray) -> float:
        return self._body_min_sdf_np(_to_np(state, STATE_DIM))

    # ------------------------------------------------------------------
    # Safety points interface  (NumPy, for CFS/QP)
    # ------------------------------------------------------------------
    def _arm_mid_pos_np(self, x: np.ndarray, side: str) -> np.ndarray:
        """World position of arm midpoint."""
        a_idx = _S_AL if side == "L" else _S_AR
        a_tuck = float(x[a_idx])
        reach = _arm_reach(a_tuck) * 0.5
        heading = float(x[_S_PSI]) + float(x[_S_PSI_T])
        sign = 1.0 if side == "L" else -1.0
        local = np.array([0.0, sign * reach], dtype=np.float32)
        c, s = np.cos(heading), np.sin(heading)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        return self._torso_pos_np(x) + R @ local

    def get_safety_points(self, state: np.ndarray) -> np.ndarray:
        """Return (5, 2): torso, arm_L_tip, arm_R_tip, arm_L_mid, arm_R_mid."""
        x = _to_np(state, STATE_DIM)
        return np.stack([
            self._torso_pos_np(x),
            self._arm_pos_np(x, "L"),
            self._arm_pos_np(x, "R"),
            self._arm_mid_pos_np(x, "L"),
            self._arm_mid_pos_np(x, "R"),
        ], axis=0).astype(np.float32)

    def get_jacobian_safety_points_action(self, state: np.ndarray) -> np.ndarray:
        zero_u = np.zeros(ACT_DIM, dtype=np.float32)
        return self.get_jacobian_safety_points_action_local(state, zero_u)

    def get_jacobian_safety_points_action_local(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        x = _to_np(state, STATE_DIM)
        u = _to_np(action, ACT_DIM)
        eps = 1e-3
        J = np.zeros((5, 2, ACT_DIM), dtype=np.float32)
        for i in range(ACT_DIM):
            du = np.zeros(ACT_DIM, dtype=np.float32)
            du[i] = eps
            pp = self.get_safety_points(self.transition(x, u + du))
            pm = self.get_safety_points(self.transition(x, u - du))
            J[:, :, i] = (pp - pm) / (2.0 * eps)
        return J.astype(np.float32)

    # ==================================================================
    # JAX methods
    # ==================================================================
    def jax_transition(self, state: Any, action: Any) -> Any:
        return self.jax_model_transition(state, action)

    def jax_model_transition(self, state: Any, action: Any) -> Any:
        if jnp is None:
            raise RuntimeError("JAX required for jax_model_transition.")
        x = jnp.asarray(state, dtype=jnp.float32).reshape(-1)[:STATE_DIM]
        u_raw = jnp.asarray(action, dtype=jnp.float32).reshape(-1)[:ACT_DIM]
        u_lo = jnp.asarray(self._u_lo, dtype=jnp.float32)
        u_hi = jnp.asarray(self._u_hi, dtype=jnp.float32)
        u = jnp.clip(u_raw, u_lo, u_hi)
        dt = jnp.asarray(self.dt, dtype=jnp.float32)

        psi = x[_S_PSI]
        c = jnp.cos(psi)
        s = jnp.sin(psi)
        vx_cmd = u[_A_VX]
        vy_cmd = u[_A_VY]

        nx = x[_S_X] + dt * (vx_cmd * c - vy_cmd * s)
        ny = x[_S_Y] + dt * (vx_cmd * s + vy_cmd * c)
        npsi = jnp.arctan2(jnp.sin(psi + dt * u[_A_OM]), jnp.cos(psi + dt * u[_A_OM]))
        nh = jnp.clip(x[_S_H] + dt * u[_A_HD], H_MIN, H_MAX)
        npt = jnp.clip(x[_S_PSI_T] + dt * u[_A_PTD], -self.psi_torso_max, self.psi_torso_max)
        nal = jnp.clip(x[_S_AL] + dt * u[_A_ADL], 0.0, 1.0)
        nar = jnp.clip(x[_S_AR] + dt * u[_A_ADR], 0.0, 1.0)
        npl = jnp.clip(x[_S_PL] + dt * u[_A_PDL], -1.0, 1.0)
        npr = jnp.clip(x[_S_PR] + dt * u[_A_PDR], -1.0, 1.0)

        out = jnp.zeros(STATE_DIM, dtype=jnp.float32)
        out = out.at[_S_X].set(nx)
        out = out.at[_S_Y].set(ny)
        out = out.at[_S_PSI].set(npsi)
        out = out.at[_S_H].set(nh)
        out = out.at[_S_PSI_T].set(npt)
        out = out.at[_S_AL].set(nal)
        out = out.at[_S_AR].set(nar)
        out = out.at[_S_PL].set(npl)
        out = out.at[_S_PR].set(npr)
        out = out.at[_S_VX].set(vx_cmd)
        out = out.at[_S_VY].set(vy_cmd)
        out = out.at[_S_OM].set(u[_A_OM])
        out = out.at[_S_HD].set(u[_A_HD])
        out = out.at[_S_PTD].set(u[_A_PTD])
        return out

    # ------------------------------------------------------------------
    # JAX SDF helpers
    # ------------------------------------------------------------------
    def _jax_box_sdf_2d(self, px: Any, py: Any, xmin: Any, xmax: Any, ymin: Any, ymax: Any) -> Any:
        """Signed distance from (px,py) to exterior of AABB. Positive outside."""
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        hx = 0.5 * (xmax - xmin)
        hy = 0.5 * (ymax - ymin)
        dx = jnp.abs(px - cx) - hx
        dy = jnp.abs(py - cy) - hy
        outside = jnp.sqrt(jnp.maximum(dx, 0.0) ** 2 + jnp.maximum(dy, 0.0) ** 2 + 1e-12)
        inside = jnp.minimum(jnp.maximum(dx, dy), 0.0)
        return outside + inside

    def _jax_z_overlap(self, h: Any, z_lo: Any, z_hi: Any) -> Any:
        body_lo = h - BODY_HALF_H
        body_hi = h + BODY_HALF_H
        overlap = jnp.minimum(body_hi, z_hi) - jnp.maximum(body_lo, z_lo)
        return jnp.clip(overlap / (2.0 * BODY_HALF_H), 0.0, 1.0)

    def _jax_ellipse_radius(self, delta_phi: Any) -> Any:
        cd = jnp.cos(delta_phi)
        sd = jnp.sin(delta_phi)
        return 1.0 / jnp.sqrt((cd / TORSO_A) ** 2 + (sd / TORSO_B) ** 2 + 1e-12)

    def _jax_arm_reach(self, a_tuck: Any) -> Any:
        return ARM_REACH_OPEN + (ARM_REACH_TUCKED - ARM_REACH_OPEN) * jnp.clip(a_tuck, 0.0, 1.0)

    def _jax_arm_pos(self, x: Any, side: str) -> Any:
        a_idx = _S_AL if side == "L" else _S_AR
        a_tuck = x[a_idx]
        reach = self._jax_arm_reach(a_tuck)
        heading = x[_S_PSI] + x[_S_PSI_T]
        sign = 1.0 if side == "L" else -1.0
        local_x = jnp.asarray(0.0, dtype=jnp.float32)
        local_y = sign * reach
        c = jnp.cos(heading)
        s = jnp.sin(heading)
        wx = x[_S_X] + c * local_x - s * local_y
        wy = x[_S_Y] + s * local_x + c * local_y
        return jnp.stack([wx, wy])

    def _jax_torso_sdf_single_obs(self, x: Any, i: int) -> Any:
        """SDF of torso ellipse to obstacle *i*, height-gated."""
        px, py = x[_S_X], x[_S_Y]
        h = x[_S_H]
        heading = x[_S_PSI] + x[_S_PSI_T]
        ox = 0.5 * (self._obs_xmin[i] + self._obs_xmax[i])
        oy = 0.5 * (self._obs_ymin[i] + self._obs_ymax[i])
        delta_phi = jnp.arctan2(oy - py, ox - px) - heading
        a_eff = TORSO_A + TORSO_CROUCH_EXTRA * jnp.maximum(0.0, H_NOMINAL - h)
        cd = jnp.cos(delta_phi)
        sd = jnp.sin(delta_phi)
        r_eff = 1.0 / jnp.sqrt((cd / a_eff) ** 2 + (sd / TORSO_B) ** 2 + 1e-12)
        d_box = self._jax_box_sdf_2d(px, py,
                                      self._obs_xmin[i], self._obs_xmax[i],
                                      self._obs_ymin[i], self._obs_ymax[i])
        z_w = self._jax_z_overlap(h, self._obs_zmin[i], self._obs_zmax[i])
        return jnp.where(z_w < 1e-4, jnp.asarray(1e6, dtype=jnp.float32),
                         (d_box - r_eff) / jnp.maximum(z_w, 1e-6))

    def _jax_arm_sdf_single_obs(self, x: Any, side: str, i: int) -> Any:
        ap = self._jax_arm_pos(x, side)
        d_box = self._jax_box_sdf_2d(ap[0], ap[1],
                                      self._obs_xmin[i], self._obs_xmax[i],
                                      self._obs_ymin[i], self._obs_ymax[i])
        z_w = self._jax_z_overlap(x[_S_H], self._obs_zmin[i], self._obs_zmax[i])
        return jnp.where(z_w < 1e-4, jnp.asarray(1e6, dtype=jnp.float32),
                         (d_box - ARM_RADIUS) / jnp.maximum(z_w, 1e-6))

    def _jax_wall_sdf(self, x: Any) -> Any:
        """Min SDF to corridor walls for all bodies."""
        w_lo = jnp.asarray(self.scene.wall_y_min, dtype=jnp.float32)
        w_hi = jnp.asarray(self.scene.wall_y_max, dtype=jnp.float32)
        h = x[_S_H]
        a_eff = TORSO_A + TORSO_CROUCH_EXTRA * jnp.maximum(0.0, H_NOMINAL - h)
        d_torso_lo = x[_S_Y] - w_lo - a_eff
        d_torso_hi = w_hi - x[_S_Y] - a_eff
        d_min = jnp.minimum(d_torso_lo, d_torso_hi)
        for side in ("L", "R"):
            ap = self._jax_arm_pos(x, side)
            d_min = jnp.minimum(d_min, ap[1] - w_lo - ARM_RADIUS)
            d_min = jnp.minimum(d_min, w_hi - ap[1] - ARM_RADIUS)
        return d_min

    def _jax_body_min_sdf(self, x: Any) -> Any:
        """Minimum SDF across all bodies, all obstacles, and walls."""
        d_min = self._jax_wall_sdf(x)
        for i in range(self._n_obs):
            d_min = jnp.minimum(d_min, self._jax_torso_sdf_single_obs(x, i))
            d_min = jnp.minimum(d_min, self._jax_arm_sdf_single_obs(x, "L", i))
            d_min = jnp.minimum(d_min, self._jax_arm_sdf_single_obs(x, "R", i))
        return d_min

    def _jax_obs_point_sdf(self, px: Any, py: Any, is_sphere: Any,
                           s_cx: Any, s_cy: Any, s_r: Any,
                           b_cx: Any, b_cy: Any, b_hx: Any, b_hy: Any,
                           is_qc: Any = None, qc_clip: Any = None) -> Any:
        """Per-obstacle SDF from a point, dispatching box/sphere/qc (vectorised)."""
        # Box SDF
        dx = jnp.abs(px - b_cx) - b_hx
        dy = jnp.abs(py - b_cy) - b_hy
        out_box = jnp.sqrt(jnp.maximum(dx, 0.0) ** 2 + jnp.maximum(dy, 0.0) ** 2 + 1e-12)
        in_box = jnp.minimum(jnp.maximum(dx, dy), 0.0)
        box_sdf = out_box + in_box
        # Sphere SDF
        sph_sdf = jnp.sqrt((px - s_cx) ** 2 + (py - s_cy) ** 2 + 1e-12) - s_r
        base = jnp.where(is_sphere > 0.5, sph_sdf, box_sdf)
        # Quarter-circle SDF: max(circle, x-clip, y-clip)
        if is_qc is not None:
            qc_circle = jnp.sqrt((px - s_cx) ** 2 + (py - s_cy) ** 2 + 1e-12) - s_r
            qc_x_hp = qc_clip * (px - s_cx)
            qc_y_hp = jnp.where(s_cy > 0, py - s_cy, s_cy - py)
            qc_sdf = jnp.maximum(qc_circle, jnp.maximum(qc_x_hp, qc_y_hp))
            base = jnp.where(is_qc > 0.5, qc_sdf, base)
        return base

    def _jax_body_min_sdf_vectorised(self, x: Any) -> Any:
        """Vectorised SDF over all obstacles using jnp operations."""
        if self._n_obs == 0:
            return self._jax_wall_sdf(x)

        obs_xmin = jnp.asarray(self._obs_xmin, dtype=jnp.float32)
        obs_xmax = jnp.asarray(self._obs_xmax, dtype=jnp.float32)
        obs_ymin = jnp.asarray(self._obs_ymin, dtype=jnp.float32)
        obs_ymax = jnp.asarray(self._obs_ymax, dtype=jnp.float32)
        obs_zmin = jnp.asarray(self._obs_zmin, dtype=jnp.float32)
        obs_zmax = jnp.asarray(self._obs_zmax, dtype=jnp.float32)
        is_sphere = jnp.asarray(self._obs_is_sphere, dtype=jnp.float32)
        sph_cx = jnp.asarray(self._obs_cx, dtype=jnp.float32)
        sph_cy = jnp.asarray(self._obs_cy, dtype=jnp.float32)
        sph_r = jnp.asarray(self._obs_radius, dtype=jnp.float32)
        is_qc = jnp.asarray(self._obs_is_qc, dtype=jnp.float32)
        qc_clip = jnp.asarray(self._obs_qc_clip_sign, dtype=jnp.float32)

        px, py = x[_S_X], x[_S_Y]
        h = x[_S_H]
        heading = x[_S_PSI] + x[_S_PSI_T]

        # z-overlap for all obstacles (N,)
        body_lo = h - BODY_HALF_H
        body_hi = h + BODY_HALF_H
        z_overlap = jnp.clip(
            (jnp.minimum(body_hi, obs_zmax) - jnp.maximum(body_lo, obs_zmin)) / (2.0 * BODY_HALF_H),
            0.0, 1.0,
        )
        z_active = z_overlap > 1e-4
        z_safe = jnp.maximum(z_overlap, 1e-6)

        # Obstacle centres (used for both box and ellipse bearing).
        obs_cx = 0.5 * (obs_xmin + obs_xmax)
        obs_cy = 0.5 * (obs_ymin + obs_ymax)
        obs_hx = 0.5 * (obs_xmax - obs_xmin)
        obs_hy = 0.5 * (obs_ymax - obs_ymin)

        # Unified point SDF for torso centre (N,)
        point_sdf_torso = self._jax_obs_point_sdf(
            px, py, is_sphere, sph_cx, sph_cy, sph_r, obs_cx, obs_cy, obs_hx, obs_hy,
            is_qc=is_qc, qc_clip=qc_clip)

        # Torso ellipse effective radius per obstacle (N,)
        delta_phi = jnp.arctan2(obs_cy - py, obs_cx - px) - heading
        a_eff = TORSO_A + TORSO_CROUCH_EXTRA * jnp.maximum(0.0, H_NOMINAL - h)
        r_eff = 1.0 / jnp.sqrt((jnp.cos(delta_phi) / a_eff) ** 2 + (jnp.sin(delta_phi) / TORSO_B) ** 2 + 1e-12)
        torso_sdf = jnp.where(z_active, (point_sdf_torso - r_eff) / z_safe, 1e6)

        # Arms
        arm_sdfs = []
        for side in ("L", "R"):
            ap = self._jax_arm_pos(x, side)
            point_sdf_arm = self._jax_obs_point_sdf(
                ap[0], ap[1], is_sphere, sph_cx, sph_cy, sph_r, obs_cx, obs_cy, obs_hx, obs_hy,
                is_qc=is_qc, qc_clip=qc_clip)
            arm_sdf = jnp.where(z_active, (point_sdf_arm - ARM_RADIUS) / z_safe, 1e6)
            arm_sdfs.append(arm_sdf)

        all_sdfs = jnp.concatenate([torso_sdf, arm_sdfs[0], arm_sdfs[1]])
        d_min_obs = jnp.min(all_sdfs)
        d_min_wall = self._jax_wall_sdf(x)
        return jnp.minimum(d_min_obs, d_min_wall)

    # ------------------------------------------------------------------
    # JAX safety points (for CBF / CFS)
    # ------------------------------------------------------------------
    def _jax_arm_mid_pos(self, x: Any, side: str) -> Any:
        """World position of arm midpoint (halfway between torso and tip)."""
        a_idx = _S_AL if side == "L" else _S_AR
        a_tuck = x[a_idx]
        reach = self._jax_arm_reach(a_tuck)
        heading = x[_S_PSI] + x[_S_PSI_T]
        sign = 1.0 if side == "L" else -1.0
        half_reach = 0.5 * reach
        local_x = jnp.asarray(0.0, dtype=jnp.float32)
        local_y = sign * half_reach
        c = jnp.cos(heading)
        s = jnp.sin(heading)
        wx = x[_S_X] + c * local_x - s * local_y
        wy = x[_S_Y] + s * local_x + c * local_y
        return jnp.stack([wx, wy])

    def jax_safety_points(self, state: Any) -> Any:
        """Return (5, 2): torso, arm_L_tip, arm_R_tip, arm_L_mid, arm_R_mid."""
        if jnp is None:
            raise RuntimeError("JAX required.")
        x = jnp.asarray(state, dtype=jnp.float32).reshape(-1)[:STATE_DIM]
        torso = jnp.stack([x[_S_X], x[_S_Y]])
        arm_L = self._jax_arm_pos(x, "L")
        arm_R = self._jax_arm_pos(x, "R")
        arm_L_mid = self._jax_arm_mid_pos(x, "L")
        arm_R_mid = self._jax_arm_mid_pos(x, "R")
        return jnp.stack([torso, arm_L, arm_R, arm_L_mid, arm_R_mid], axis=0)

    def jax_cfs_safety_point_weights(self, state: Any) -> Any:
        """All 5 bodies equally weighted."""
        return jnp.ones(5, dtype=jnp.float32)

    def jax_cfs_alm_g_plus_from_state(self, state: Any, clearance: Any) -> Any:
        """Structured iALM constraint violation g(z) for corridor avoidance.

        Computes max [g_i(z)]+ across 4 constraint groups:
          A. Body envelope clearance (per body part, height-gated)
          B. Squeeze feasibility (W_eff vs W_free)
          C. Posture bounds
          D. Wall clearance

        Returns scalar: max violation across all constraints.
        This drives the ALM penalty  λ·[g]+ + ρ/2·[g]+²  in the
        augmented reward, making diffusion samples avoid unsafe states.
        """
        if jnp is None:
            raise RuntimeError("JAX required.")
        x = jnp.asarray(state, dtype=jnp.float32).reshape(-1)[:STATE_DIM]
        c = jnp.asarray(clearance, dtype=jnp.float32).reshape(-1)[0]
        h = x[_S_H]

        violations = []

        # ── Group A: body envelope clearance (height-gated) ──
        # 5 body parts: torso, arm_L_tip, arm_R_tip, arm_L_mid, arm_R_mid.
        a_eff = TORSO_A + TORSO_CROUCH_EXTRA * jnp.maximum(0.0, H_NOMINAL - h)
        body_radii = jnp.array([a_eff, ARM_RADIUS, ARM_RADIUS, ARM_RADIUS, ARM_RADIUS], dtype=jnp.float32)
        pts5 = self.jax_safety_points(x)  # (5, 2)

        g_scale = jnp.asarray(self.scene.corridor_width / 2.0, dtype=jnp.float32)

        for i in range(5):
            sdf_val = self._jax_height_aware_sdf_at_point(pts5[i, 0], pts5[i, 1], h)
            d_margin = sdf_val - body_radii[i]
            g_clr = jnp.maximum(0.0, -d_margin) / g_scale
            violations.append(g_clr)

        # ── Group B: squeeze feasibility ──
        w_eff = self.jax_w_eff(x)
        w_free = self.jax_w_free(x)
        g_sq = jnp.maximum(0.0, (w_eff - w_free)) / g_scale
        violations.append(g_sq)

        # ── Group C: posture bounds ──
        g_h_lo = H_MIN - h
        g_h_hi = h - H_MAX
        g_torso = jnp.abs(x[_S_PSI_T]) - self.psi_torso_max
        violations.append(jnp.maximum(0.0, g_h_lo))
        violations.append(jnp.maximum(0.0, g_h_hi))
        violations.append(jnp.maximum(0.0, g_torso))

        # ── Group D: wall clearance (normalised by g_scale) ──
        w_lo = jnp.asarray(self.scene.wall_y_min, dtype=jnp.float32)
        w_hi = jnp.asarray(self.scene.wall_y_max, dtype=jnp.float32)
        for i in range(5):
            d_wlo = pts5[i, 1] - w_lo - body_radii[i]
            d_whi = w_hi - pts5[i, 1] - body_radii[i]
            violations.append(jnp.maximum(0.0, -d_wlo) / g_scale)
            violations.append(jnp.maximum(0.0, -d_whi) / g_scale)

        return jnp.max(jnp.stack(violations))

    # ------------------------------------------------------------------
    # Effective width / squeeze feasibility (JAX)
    # ------------------------------------------------------------------
    def jax_w_eff(self, state: Any) -> Any:
        """Effective lateral width of the humanoid given current posture.

        Returns a scalar proxy for how wide the body is, accounting for
        torso yaw (sidestep narrows the profile) and arm tuck.
        """
        x = jnp.asarray(state, dtype=jnp.float32).reshape(-1)[:STATE_DIM]
        psi_t = jnp.abs(x[_S_PSI_T])
        a_sum = x[_S_AL] + x[_S_AR]
        return W_EFF_W0 - W_EFF_ALPHA_TORSO * psi_t - W_EFF_ALPHA_ARM * a_sum

    def jax_w_free(self, state: Any) -> Any:
        """Local free corridor width at the robot's x-position.

        Only considers **full-height** obstacles (z_min <= 0.05 AND z_max >= 1.8)
        that laterally squeeze the corridor. Height-gated obstacles (low bars,
        aerial spheres) don't reduce W_free — they are handled by Group A
        clearance constraints in the iALM g(z) vector.
        """
        x = jnp.asarray(state, dtype=jnp.float32).reshape(-1)[:STATE_DIM]
        px = x[_S_X]
        w_corridor = jnp.asarray(self.scene.corridor_width, dtype=jnp.float32)

        if self._n_obs == 0:
            return w_corridor

        obs_xmin = jnp.asarray(self._obs_xmin, dtype=jnp.float32)
        obs_xmax = jnp.asarray(self._obs_xmax, dtype=jnp.float32)
        obs_ymin = jnp.asarray(self._obs_ymin, dtype=jnp.float32)
        obs_ymax = jnp.asarray(self._obs_ymax, dtype=jnp.float32)
        obs_zmin = jnp.asarray(self._obs_zmin, dtype=jnp.float32)
        obs_zmax = jnp.asarray(self._obs_zmax, dtype=jnp.float32)

        # x-overlap: does the robot's x fall inside the obstacle's x range?
        x_inside = (px >= obs_xmin) & (px <= obs_xmax)

        # Only full-height obstacles affect W_free (walls/protrusions/squeeze).
        is_full_height = (obs_zmin <= 0.05) & (obs_zmax >= 1.8)

        active = x_inside & is_full_height

        # For each active obstacle, compute how much corridor width it eats.
        # Left protrusion (y_min > 0): eats from +side → reduces gap by (wall_hi - y_min).
        # Right protrusion (y_max < 0): eats from -side → reduces gap by (y_max - wall_lo).
        # General: subtract obstacle lateral extent from corridor.
        hw = w_corridor / 2.0
        # Width eaten from the positive side.
        eaten_pos = jnp.where(active & (obs_ymax > 0), jnp.maximum(0.0, hw - obs_ymin), 0.0)
        # Width eaten from the negative side.
        eaten_neg = jnp.where(active & (obs_ymin < 0), jnp.maximum(0.0, obs_ymax + hw), 0.0)

        total_eaten = jnp.maximum(jnp.max(eaten_pos), 0.0) + jnp.maximum(jnp.max(eaten_neg), 0.0)
        return jnp.maximum(w_corridor - total_eaten, 0.0)

    def jax_squeeze_violation(self, state: Any) -> Any:
        """g_sq = W_eff(z) - W_free(x,y,h).  Positive → body too wide for gap."""
        return self.jax_w_eff(state) - self.jax_w_free(state)

    def jax_jacobian_safety_points_action(self, state: Any) -> Any:
        if jax is None:
            raise RuntimeError("JAX required.")
        x = jnp.asarray(state, dtype=jnp.float32).reshape(-1)[:STATE_DIM]
        u0 = jnp.zeros(ACT_DIM, dtype=jnp.float32)

        def pts_next(u):
            return self.jax_safety_points(self.jax_model_transition(x, u))

        return jax.jacfwd(pts_next)(u0).astype(jnp.float32)

    def jax_jacobian_safety_points_action_local(self, state: Any, action: Any) -> Any:
        if jax is None:
            raise RuntimeError("JAX required.")
        x = jnp.asarray(state, dtype=jnp.float32).reshape(-1)[:STATE_DIM]
        u0 = jnp.asarray(action, dtype=jnp.float32).reshape(-1)[:ACT_DIM]

        def pts_next(u):
            return self.jax_safety_points(self.jax_model_transition(x, u))

        return jax.jacfwd(pts_next)(u0).astype(jnp.float32)

    # ------------------------------------------------------------------
    # JAX cost / terminal
    # ------------------------------------------------------------------
    def jax_cost(self, state: Any) -> Any:
        if jnp is None:
            raise RuntimeError("JAX required.")
        x = jnp.asarray(state, dtype=jnp.float32).reshape(-1)[:STATE_DIM]
        goal = jnp.asarray(self.target, dtype=jnp.float32)
        goal_err = jnp.sum((x[:2] - goal) ** 2)
        center_err = x[_S_Y] ** 2
        arm_err = x[_S_AL] ** 2 + x[_S_AR] ** 2
        torso_err = x[_S_PSI_T] ** 2
        height_err = (x[_S_H] - H_NOMINAL) ** 2
        return goal_err + 2.0 * center_err + arm_err + torso_err + 2.0 * height_err

    def terminal_distance_jax(self, state: Any) -> Any:
        if jnp is None:
            raise RuntimeError("JAX required.")
        x = jnp.asarray(state, dtype=jnp.float32).reshape(-1)[:STATE_DIM]
        goal = jnp.asarray(self.target, dtype=jnp.float32)
        pos_err = jnp.linalg.norm(x[:2] - goal)
        # Penalise residual arm tuck / torso twist at goal.
        posture_err = 0.3 * (x[_S_AL] ** 2 + x[_S_AR] ** 2 + x[_S_PSI_T] ** 2)
        return pos_err + posture_err

    def jax_jacobian_xy(self):
        """Return constant Jacobian for CFS QP action-to-position lift.

        CFS standard path computes: lhs = dt * J_xy @ grad_sdf @ cumsum(u)
        For single integrator: J_xy = I (action=acceleration, position=dt*cumsum(a))
        For velocity-rate: action=velocity, position=dt*v (no cumsum needed).
        But CFS always does cumsum internally, so we use I to match the
        expected scale — the dt factor is already in CFS's lhs formula.

        Returns (2, act_dim) JAX array.
        """
        J = jnp.zeros((2, ACT_DIM), dtype=jnp.float32)
        J = J.at[0, _A_VX].set(1.0)
        J = J.at[1, _A_VY].set(1.0)
        return J

    # ------------------------------------------------------------------
    # JAX scene SDF (for CFS QP)
    # ------------------------------------------------------------------
    def _jax_scene_qp_sdf(self, pt: Any) -> Any:
        """Signed distance for a single 2D point (positive = safe).
        Used by CFS QP constraints; ignores z-gating (conservative)."""
        if jnp is None:
            raise RuntimeError("JAX required.")
        p = jnp.asarray(pt, dtype=jnp.float32).reshape(-1)[:2]
        if self._n_obs == 0:
            w_lo = jnp.asarray(self.scene.wall_y_min, dtype=jnp.float32)
            w_hi = jnp.asarray(self.scene.wall_y_max, dtype=jnp.float32)
            return jnp.minimum(p[1] - w_lo, w_hi - p[1])

        obs_xmin = jnp.asarray(self._obs_xmin, dtype=jnp.float32)
        obs_xmax = jnp.asarray(self._obs_xmax, dtype=jnp.float32)
        obs_ymin = jnp.asarray(self._obs_ymin, dtype=jnp.float32)
        obs_ymax = jnp.asarray(self._obs_ymax, dtype=jnp.float32)
        obs_cx = 0.5 * (obs_xmin + obs_xmax)
        obs_cy = 0.5 * (obs_ymin + obs_ymax)
        obs_hx = 0.5 * (obs_xmax - obs_xmin)
        obs_hy = 0.5 * (obs_ymax - obs_ymin)
        is_sphere = jnp.asarray(self._obs_is_sphere, dtype=jnp.float32)
        sph_cx = jnp.asarray(self._obs_cx, dtype=jnp.float32)
        sph_cy = jnp.asarray(self._obs_cy, dtype=jnp.float32)
        sph_r = jnp.asarray(self._obs_radius, dtype=jnp.float32)
        is_qc = jnp.asarray(self._obs_is_qc, dtype=jnp.float32)
        qc_clip = jnp.asarray(self._obs_qc_clip_sign, dtype=jnp.float32)
        obs_sdfs = self._jax_obs_point_sdf(
            p[0], p[1], is_sphere, sph_cx, sph_cy, sph_r, obs_cx, obs_cy, obs_hx, obs_hy,
            is_qc=is_qc, qc_clip=qc_clip)
        min_obs_sdf = jnp.min(obs_sdfs)
        w_lo = jnp.asarray(self.scene.wall_y_min, dtype=jnp.float32)
        w_hi = jnp.asarray(self.scene.wall_y_max, dtype=jnp.float32)
        wall_sdf = jnp.minimum(p[1] - w_lo, w_hi - p[1])
        return jnp.minimum(min_obs_sdf, wall_sdf)

    def _jax_height_aware_sdf_at_point(self, px: Any, py: Any, h: Any) -> Any:
        """Height-gated SDF from a 2D point to all obstacles + walls.

        Returns a scalar: min SDF across all obstacles, where obstacles
        outside the body's vertical extent are ignored (SDF = +inf).
        """
        w_lo = jnp.asarray(self.scene.wall_y_min, dtype=jnp.float32)
        w_hi = jnp.asarray(self.scene.wall_y_max, dtype=jnp.float32)
        d_min = jnp.minimum(py - w_lo, w_hi - py)

        if self._n_obs == 0:
            return d_min

        obs_xmin = jnp.asarray(self._obs_xmin, dtype=jnp.float32)
        obs_xmax = jnp.asarray(self._obs_xmax, dtype=jnp.float32)
        obs_ymin = jnp.asarray(self._obs_ymin, dtype=jnp.float32)
        obs_ymax = jnp.asarray(self._obs_ymax, dtype=jnp.float32)
        obs_zmin = jnp.asarray(self._obs_zmin, dtype=jnp.float32)
        obs_zmax = jnp.asarray(self._obs_zmax, dtype=jnp.float32)
        is_sphere = jnp.asarray(self._obs_is_sphere, dtype=jnp.float32)
        sph_cx = jnp.asarray(self._obs_cx, dtype=jnp.float32)
        sph_cy = jnp.asarray(self._obs_cy, dtype=jnp.float32)
        sph_r = jnp.asarray(self._obs_radius, dtype=jnp.float32)
        is_qc = jnp.asarray(self._obs_is_qc, dtype=jnp.float32)
        qc_clip = jnp.asarray(self._obs_qc_clip_sign, dtype=jnp.float32)
        obs_cx = 0.5 * (obs_xmin + obs_xmax)
        obs_cy = 0.5 * (obs_ymin + obs_ymax)
        obs_hx = 0.5 * (obs_xmax - obs_xmin)
        obs_hy = 0.5 * (obs_ymax - obs_ymin)

        # z-overlap gating
        body_lo = h - BODY_HALF_H
        body_hi = h + BODY_HALF_H
        z_ov = jnp.clip(
            (jnp.minimum(body_hi, obs_zmax) - jnp.maximum(body_lo, obs_zmin)) / (2.0 * BODY_HALF_H),
            0.0, 1.0)
        z_active = z_ov > 1e-4
        z_safe = jnp.maximum(z_ov, 1e-6)

        point_sdf = self._jax_obs_point_sdf(
            px, py, is_sphere, sph_cx, sph_cy, sph_r, obs_cx, obs_cy, obs_hx, obs_hy,
            is_qc=is_qc, qc_clip=qc_clip)
        gated_sdf = jnp.where(z_active, point_sdf / z_safe, 1e6)
        return jnp.minimum(d_min, jnp.min(gated_sdf))

    def _jax_single_obs_sdf_height_gated(self, px: Any, py: Any, h: Any, obs_idx: int) -> Any:
        """Height-gated SDF from a point to a single obstacle."""
        obs_xmin = jnp.asarray(self._obs_xmin, dtype=jnp.float32)
        obs_xmax = jnp.asarray(self._obs_xmax, dtype=jnp.float32)
        obs_ymin = jnp.asarray(self._obs_ymin, dtype=jnp.float32)
        obs_ymax = jnp.asarray(self._obs_ymax, dtype=jnp.float32)
        obs_zmin = jnp.asarray(self._obs_zmin, dtype=jnp.float32)
        obs_zmax = jnp.asarray(self._obs_zmax, dtype=jnp.float32)
        is_sphere = jnp.asarray(self._obs_is_sphere, dtype=jnp.float32)
        sph_cx = jnp.asarray(self._obs_cx, dtype=jnp.float32)
        sph_cy = jnp.asarray(self._obs_cy, dtype=jnp.float32)
        sph_r = jnp.asarray(self._obs_radius, dtype=jnp.float32)
        is_qc = jnp.asarray(self._obs_is_qc, dtype=jnp.float32)
        qc_clip = jnp.asarray(self._obs_qc_clip_sign, dtype=jnp.float32)

        # z-overlap
        body_lo = h - BODY_HALF_H
        body_hi = h + BODY_HALF_H
        z_ov = jnp.clip(
            (jnp.minimum(body_hi, obs_zmax[obs_idx]) - jnp.maximum(body_lo, obs_zmin[obs_idx]))
            / (2.0 * BODY_HALF_H), 0.0, 1.0)
        z_active = z_ov > 1e-4
        z_safe = jnp.maximum(z_ov, 1e-6)

        # Point SDF
        cx = 0.5 * (obs_xmin[obs_idx] + obs_xmax[obs_idx])
        cy = 0.5 * (obs_ymin[obs_idx] + obs_ymax[obs_idx])
        hx = 0.5 * (obs_xmax[obs_idx] - obs_xmin[obs_idx])
        hy = 0.5 * (obs_ymax[obs_idx] - obs_ymin[obs_idx])
        is_s = is_sphere[obs_idx]
        # Box SDF
        dx = jnp.abs(px - cx) - hx
        dy = jnp.abs(py - cy) - hy
        box_d = jnp.sqrt(jnp.maximum(dx, 0.0)**2 + jnp.maximum(dy, 0.0)**2 + 1e-12) + jnp.minimum(jnp.maximum(dx, dy), 0.0)
        # Sphere SDF
        sph_d = jnp.sqrt((px - sph_cx[obs_idx])**2 + (py - sph_cy[obs_idx])**2 + 1e-12) - sph_r[obs_idx]
        point_sdf = jnp.where(is_s > 0.5, sph_d, box_d)
        # Quarter-circle SDF
        is_q = is_qc[obs_idx]
        qc_circle = jnp.sqrt((px - sph_cx[obs_idx])**2 + (py - sph_cy[obs_idx])**2 + 1e-12) - sph_r[obs_idx]
        qc_x_hp = qc_clip[obs_idx] * (px - sph_cx[obs_idx])
        qc_y_hp = jnp.where(sph_cy[obs_idx] > 0, py - sph_cy[obs_idx], sph_cy[obs_idx] - py)
        qc_d = jnp.maximum(qc_circle, jnp.maximum(qc_x_hp, qc_y_hp))
        point_sdf = jnp.where(is_q > 0.5, qc_d, point_sdf)

        return jnp.where(z_active, point_sdf / z_safe, 1e6)

    def jax_cfs_custom_safety_constraints(
        self,
        state_prev: Any,
        state_t: Any,
        action_ref: Any,
        pts_t: Any,
        J_pts_t: Any,
        clearance_s: Any,
        act_dim: int,
        k_select: int,
    ) -> Any:
        """Per-obstacle × per-body-part CFS constraints for corridor avoidance.

        Generates up to N_obs × 3 candidate constraints (one per obstacle per
        body part), each with height-gated SDF.  Selects top-k_select most
        violated constraints for the QP.

        Returns (A, b, valid) with shapes (k_select, act_dim), (k_select,), (k_select,).
        """
        if jnp is None:
            raise RuntimeError("JAX required.")
        _ = state_prev
        ad = int(act_dim)
        ks = int(k_select)
        if ks <= 0:
            return (
                jnp.zeros((0, ad), dtype=jnp.float32),
                jnp.zeros((0,), dtype=jnp.float32),
                jnp.zeros((0,), dtype=jnp.bool_),
            )

        x = jnp.asarray(state_t, dtype=jnp.float32).reshape(-1)[:STATE_DIM]
        h = x[_S_H]
        c_scalar = jnp.asarray(clearance_s, dtype=jnp.float32).reshape(-1)[0]
        u_ref = jnp.asarray(action_ref, dtype=jnp.float32).reshape(-1)[:ad]

        n_pts = 5  # torso, arm_L_tip, arm_R_tip, arm_L_mid, arm_R_mid
        pts5 = jnp.asarray(pts_t, dtype=jnp.float32).reshape(n_pts, 2)
        J5 = jnp.asarray(J_pts_t, dtype=jnp.float32)[:n_pts, :2, :ad]

        a_eff = TORSO_A + TORSO_CROUCH_EXTRA * jnp.maximum(0.0, H_NOMINAL - h)
        body_radii = jnp.array([a_eff, ARM_RADIUS, ARM_RADIUS, ARM_RADIUS, ARM_RADIUS], dtype=jnp.float32)

        w_lo = jnp.asarray(self.scene.wall_y_min, dtype=jnp.float32)
        w_hi = jnp.asarray(self.scene.wall_y_max, dtype=jnp.float32)

        # Fix A + D: velocity-aware dynamic clearance.
        # At high speed, CFS needs more lead time to steer away.
        # c_eff = base_clearance + conserv_margin + v_mag * dt * lookahead_steps
        v_mag = jnp.sqrt(x[_S_VX] ** 2 + x[_S_VY] ** 2 + 1e-8)
        dt_env = jnp.asarray(self.dt, dtype=jnp.float32)
        velocity_margin = v_mag * dt_env * 2.0  # 2-step lookahead
        conserv_margin = jnp.asarray(0.05, dtype=jnp.float32)
        c_eff = c_scalar + conserv_margin + velocity_margin
        # Activation threshold: start constraining well before collision.
        activation_range = c_eff + 0.30  # activate 30cm before effective clearance

        A_list = []
        b_list = []
        v_list = []

        for bi in range(n_pts):
            p = pts5[bi]
            J = J5[bi]
            r = body_radii[bi]

            # Wall bottom: sdf = p_y - w_lo - r
            d_wlo = p[1] - w_lo - r
            grad_wlo = jnp.array([0.0, 1.0], dtype=jnp.float32)
            A_wlo = jnp.matmul(jnp.transpose(J), grad_wlo)
            b_wlo = c_eff - d_wlo + jnp.dot(A_wlo, u_ref)
            v_wlo = d_wlo < activation_range

            # Wall top: sdf = w_hi - p_y - r
            d_whi = w_hi - p[1] - r
            grad_whi = jnp.array([0.0, -1.0], dtype=jnp.float32)
            A_whi = jnp.matmul(jnp.transpose(J), grad_whi)
            b_whi = c_eff - d_whi + jnp.dot(A_whi, u_ref)
            v_whi = d_whi < activation_range

            A_list.extend([A_wlo, A_whi])
            b_list.extend([b_wlo, b_whi])
            v_list.extend([v_wlo, v_whi])

            # Per-obstacle constraints (height-gated).
            for oi in range(self._n_obs):
                sdf_val = self._jax_single_obs_sdf_height_gated(p[0], p[1], h, oi)
                d_margin = sdf_val - r

                # Gradient w.r.t. body position.
                def _sdf_fn(pp, _oi=oi, _h=h):
                    return self._jax_single_obs_sdf_height_gated(pp[0], pp[1], _h, _oi)
                grad_p = jax.grad(_sdf_fn)(p)
                grad_p = jnp.where(jnp.isnan(grad_p), 0.0, grad_p)

                A_row = jnp.matmul(jnp.transpose(J), grad_p)
                b_val = c_eff - d_margin + jnp.dot(A_row, u_ref)
                is_valid = (d_margin < activation_range) & (jnp.linalg.norm(grad_p) > 1e-6)

                A_list.append(A_row)
                b_list.append(b_val)
                v_list.append(is_valid)

        # Stack all candidates: n_total = 3 * (2 walls + n_obs)
        A_all = jnp.stack(A_list, axis=0)
        b_all = jnp.stack(b_list, axis=0)
        v_all = jnp.stack(v_list, axis=0)
        n_total = A_all.shape[0]

        if ks >= n_total:
            pad = ks - n_total
            A_out = jnp.concatenate([A_all, jnp.zeros((pad, ad), dtype=jnp.float32)], axis=0)
            b_out = jnp.concatenate([b_all, jnp.full((pad,), -jnp.inf, dtype=jnp.float32)], axis=0)
            v_out = jnp.concatenate([v_all, jnp.zeros((pad,), dtype=jnp.bool_)], axis=0)
        else:
            # Select top-k most violated constraints.
            # Violation = how much clearance is missing: c_scalar - d_margin.
            violation_score = jnp.where(v_all, b_all - jnp.einsum('ij,j->i', A_all, u_ref), -1e9)
            _, top_idx = jax.lax.top_k(violation_score, ks)
            A_out = A_all[top_idx]
            b_out = b_all[top_idx]
            v_out = v_all[top_idx]

        return A_out, b_out, v_out


# ---------------------------------------------------------------------------
# Energy functional factory
# ---------------------------------------------------------------------------
def make_corridor_energy(env: HumanoidCorridor2DEnv) -> LegacyEnergyFunctional:
    """Build the LegacyEnergyFunctional for corridor obstacle avoidance."""
    goal = np.asarray(env.target, dtype=np.float32)
    start = np.asarray(env.start, dtype=np.float32)
    corridor_length = float(env.scene.corridor_length)
    H_plan = int(env.horizon)

    if jnp is not None:
        goal_j = jnp.asarray(goal, dtype=jnp.float32)
        start_j = jnp.asarray(start, dtype=jnp.float32)

        # Pre-pack obstacle arrays for vectorised JAX SDF.
        n_obs = env._n_obs
        obs_xmin_j = jnp.asarray(env._obs_xmin, dtype=jnp.float32)
        obs_xmax_j = jnp.asarray(env._obs_xmax, dtype=jnp.float32)
        obs_ymin_j = jnp.asarray(env._obs_ymin, dtype=jnp.float32)
        obs_ymax_j = jnp.asarray(env._obs_ymax, dtype=jnp.float32)
        obs_zmin_j = jnp.asarray(env._obs_zmin, dtype=jnp.float32)
        obs_zmax_j = jnp.asarray(env._obs_zmax, dtype=jnp.float32)
        is_sphere_j = jnp.asarray(env._obs_is_sphere, dtype=jnp.float32)
        is_qc_j = jnp.asarray(env._obs_is_qc, dtype=jnp.float32)
        sph_cx_j = jnp.asarray(env._obs_cx, dtype=jnp.float32)
        sph_cy_j = jnp.asarray(env._obs_cy, dtype=jnp.float32)
        sph_r_j = jnp.asarray(env._obs_radius, dtype=jnp.float32)
        qc_clip_j = jnp.asarray(env._obs_qc_clip_sign, dtype=jnp.float32)
        corridor_length_half_w = env.scene.corridor_width / 2.0
        w_lo_j = jnp.asarray(env.scene.wall_y_min, dtype=jnp.float32)
        w_hi_j = jnp.asarray(env.scene.wall_y_max, dtype=jnp.float32)

        def _point_sdf_energy(px, py):
            """Dispatched point SDF for box/sphere/qc (energy function)."""
            obs_cx = 0.5 * (obs_xmin_j + obs_xmax_j)
            obs_cy = 0.5 * (obs_ymin_j + obs_ymax_j)
            obs_hx = 0.5 * (obs_xmax_j - obs_xmin_j)
            obs_hy = 0.5 * (obs_ymax_j - obs_ymin_j)
            # Box
            dx = jnp.abs(px - obs_cx) - obs_hx
            dy = jnp.abs(py - obs_cy) - obs_hy
            box_d = jnp.sqrt(jnp.maximum(dx, 0.0) ** 2 + jnp.maximum(dy, 0.0) ** 2 + 1e-12) + jnp.minimum(jnp.maximum(dx, dy), 0.0)
            # Sphere
            sph_d = jnp.sqrt((px - sph_cx_j) ** 2 + (py - sph_cy_j) ** 2 + 1e-12) - sph_r_j
            d = jnp.where(is_sphere_j > 0.5, sph_d, box_d)
            # Quarter-circle
            qc_circle = jnp.sqrt((px - sph_cx_j) ** 2 + (py - sph_cy_j) ** 2 + 1e-12) - sph_r_j
            qc_x_hp = qc_clip_j * (px - sph_cx_j)
            qc_y_hp = jnp.where(sph_cy_j > 0, py - sph_cy_j, sph_cy_j - py)
            qc_d = jnp.maximum(qc_circle, jnp.maximum(qc_x_hp, qc_y_hp))
            d = jnp.where(is_qc_j > 0.5, qc_d, d)
            return d

        def _obs_sdf_j(px, py, h, heading):
            """Min SDF from torso + arms to all obstacles + walls (JAX)."""
            # z-overlap
            body_lo = h - BODY_HALF_H
            body_hi = h + BODY_HALF_H
            z_ov = jnp.clip(
                (jnp.minimum(body_hi, obs_zmax_j) - jnp.maximum(body_lo, obs_zmin_j)) / (2.0 * BODY_HALF_H),
                0.0, 1.0)
            z_act = z_ov > 1e-4
            z_s = jnp.maximum(z_ov, 1e-6)

            # Torso: point SDF via dispatched function
            point_t = _point_sdf_energy(px, py)
            obs_cx = 0.5 * (obs_xmin_j + obs_xmax_j)
            obs_cy = 0.5 * (obs_ymin_j + obs_ymax_j)
            dp = jnp.arctan2(obs_cy - py, obs_cx - px) - heading
            a_e = TORSO_A + TORSO_CROUCH_EXTRA * jnp.maximum(0.0, H_NOMINAL - h)
            r_e = 1.0 / jnp.sqrt((jnp.cos(dp) / a_e) ** 2 + (jnp.sin(dp) / TORSO_B) ** 2 + 1e-12)
            tsdf = jnp.where(z_act, (point_t - r_e) / z_s, 1e6)

            d = jnp.min(tsdf)
            # Wall SDF for torso
            d = jnp.minimum(d, py - w_lo_j - a_e)
            d = jnp.minimum(d, w_hi_j - py - a_e)
            return d

        def task_energy(x, u, ctx):
            st = jnp.asarray(x, dtype=jnp.float32).reshape(-1)[:STATE_DIM]
            act = jnp.asarray(u, dtype=jnp.float32).reshape(-1)[:ACT_DIM]

            pos = st[:2]
            dist_to_goal = jnp.linalg.norm(pos - goal_j + 1e-6)

            # Forward progress — scale gradient with corridor length so that
            # long corridors get the same per-metre incentive as short ones.
            corridor_span = jnp.maximum(goal_j[0] - start_j[0], 1e-6)
            progress = (st[_S_X] - start_j[0]) / corridor_span
            progress_scale = jnp.maximum(corridor_span / 4.0, 1.0)  # normalise to ~4 m baseline
            progress_reward = -0.3 * progress_scale * jnp.clip(progress, 0.0, 1.0)

            # Goal attraction — mildly scale with corridor length.
            # Cap at 2× so 4 m zones stay at 0.2, 12 m → 0.4.
            goal_scale = jnp.clip(corridor_span / 6.0, 1.0, 2.0)
            goal_err = 0.2 * goal_scale * jnp.sum((pos - goal_j) ** 2)

            # Centreline preference (mild).
            center_err = 1.0 * st[_S_Y] ** 2

            # Posture regularisation (very mild — let geometry/CFS drive posture changes).
            arm_err = 0.3 * (st[_S_AL] ** 2 + st[_S_AR] ** 2)
            torso_err = 0.3 * st[_S_PSI_T] ** 2
            height_err = 6.0 * (st[_S_H] - H_NOMINAL) ** 2

            # Velocity regularisation.
            vel_err = 0.05 * (st[_S_VX] ** 2 + st[_S_VY] ** 2 + st[_S_OM] ** 2)

            # Control smoothness.
            ctrl_err = 0.06 * jnp.sum(act ** 2)

            # (E) Bottleneck-aware squeeze: penalise W_eff > W_free,
            # reward W_eff < W_free.  Only active when W_free < W0 (near obstacles).
            w_eff = W_EFF_W0 - W_EFF_ALPHA_TORSO * jnp.abs(st[_S_PSI_T]) - W_EFF_ALPHA_ARM * (st[_S_AL] + st[_S_AR])
            if n_obs > 0:
                hw_j = jnp.asarray(float(corridor_length_half_w), dtype=jnp.float32)
                x_in = (st[_S_X] >= obs_xmin_j) & (st[_S_X] <= obs_xmax_j)
                # Only full-height obstacles squeeze the corridor laterally.
                is_full_h = (obs_zmin_j <= 0.05) & (obs_zmax_j >= 1.8)
                active_sq = x_in & is_full_h
                eaten_pos = jnp.where(active_sq & (obs_ymax_j > 0), jnp.maximum(0.0, hw_j - obs_ymin_j), 0.0)
                eaten_neg = jnp.where(active_sq & (obs_ymin_j < 0), jnp.maximum(0.0, obs_ymax_j + hw_j), 0.0)
                w_free = jnp.maximum(2.0 * hw_j - jnp.max(eaten_pos) - jnp.max(eaten_neg), 0.01)
            else:
                w_free = jnp.asarray(2.0 * float(corridor_length_half_w), dtype=jnp.float32)
            # Proximity factor: 1.0 at tight squeeze, 0.0 in open space.
            bottleneck_proximity = jnp.clip(1.0 - w_free / (2.0 * float(corridor_length_half_w)), 0.0, 1.0)
            # Squeeze cost: penalise being wider than the gap, reward being narrower.
            squeeze_gap = jnp.maximum(w_eff - w_free, 0.0)
            squeeze_err = 5.0 * bottleneck_proximity * squeeze_gap ** 2

            return goal_err + center_err + arm_err + torso_err + height_err + vel_err + ctrl_err + progress_reward + squeeze_err

        def obstacle_energy(x, u, ctx):
            return jnp.asarray(0.0, dtype=jnp.float32)

        def bounds_energy(x, u, ctx):
            st = jnp.asarray(x, dtype=jnp.float32).reshape(-1)[:STATE_DIM]
            h_lo_v = jnp.maximum(0.0, H_MIN - st[_S_H]) ** 2
            h_hi_v = jnp.maximum(0.0, st[_S_H] - H_MAX) ** 2
            a_lo = jnp.maximum(0.0, -st[_S_AL]) ** 2 + jnp.maximum(0.0, -st[_S_AR]) ** 2
            a_hi = jnp.maximum(0.0, st[_S_AL] - 1.0) ** 2 + jnp.maximum(0.0, st[_S_AR] - 1.0) ** 2
            p_v = jnp.maximum(0.0, jnp.abs(st[_S_PL]) - 1.0) ** 2 + jnp.maximum(0.0, jnp.abs(st[_S_PR]) - 1.0) ** 2
            t_v = jnp.maximum(0.0, jnp.abs(st[_S_PSI_T]) - 1.57) ** 2
            return 10.0 * (h_lo_v + h_hi_v + a_lo + a_hi + p_v + t_v)

        return LegacyEnergyFunctional({
            "task": EnergyTerm(task_energy, 1.0),
            "obstacle": EnergyTerm(obstacle_energy, 1.0),
            "bounds": EnergyTerm(bounds_energy, 1.0),
        })

    # Fallback: NumPy-only energy (rare).
    def task_energy_np(x, u, ctx):
        st = np.asarray(x, dtype=np.float32).ravel()[:STATE_DIM]
        act = np.asarray(u, dtype=np.float32).ravel()[:ACT_DIM]
        pos = st[:2]
        goal_err = float(np.sum((pos - goal) ** 2))
        center_err = 2.0 * float(st[_S_Y] ** 2)
        arm_err = float(st[_S_AL] ** 2 + st[_S_AR] ** 2)
        torso_err = 1.5 * float(st[_S_PSI_T] ** 2)
        height_err = 2.0 * float((st[_S_H] - H_NOMINAL) ** 2)
        ctrl_err = 0.08 * float(np.sum(act ** 2))
        return goal_err + center_err + arm_err + torso_err + height_err + ctrl_err

    return LegacyEnergyFunctional({
        "task": EnergyTerm(task_energy_np, 1.0),
    })
