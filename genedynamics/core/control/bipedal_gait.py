"""Phase-driven bipedal stepping gait — the leg-impedance balance substrate for a humanoid
manipulation pi_low (whole-body impedance = hand Cartesian impedance + leg joint impedance).

Why a gait at all: a STATIC stance cannot balance a humanoid push. The push (and the arm
reaching forward) shifts the CoM ahead of the feet; the ankle torque saturates (~40 Nm vs a
~50 Nm tipping moment) and the robot tips. The fix is to STEP — reposition the swing foot under
a CoM *capture point* so the support follows the CoM (dynamic balance). DIAL avoids the issue by
optimizing joint targets across their configured ranges, not by using this CPG.
This reusable reference generator is therefore a different low-level mechanism;
its physical walking feasibility must be validated for each embodiment and load.

What this is: a small CPG. Two legs run 180 deg out of phase; each cycle a leg lifts (knee) and
swings its hip from back to front, and the per-leg hip target is biased by the *capture term*
(CoM position + a lead on CoM velocity) so the foot lands under where the CoM is heading, plus a
forward bias from the base velocity command ``v_cmd`` (0 = step in place for a standing push,
>0 = walk). It returns leg joint-position-target OFFSETS (from the default pose); a joint-impedance
(PD) law in the env tracks them — that PD is the leg's joint-space impedance.

Robot-agnostic: constructed from the per-leg sagittal joint indices (hip_pitch, knee, ankle_pitch)
into the actuated-joint vector; nothing H1/G1-specific lives here. Pure jax (jit/vmap-safe).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax.numpy as jnp


@dataclass(frozen=True)
class GaitParams:
    cadence: float = 1.6        # step cycles per second (gait frequency)
    swing_frac: float = 0.45    # fraction of the cycle a leg is in swing (rest = stance)
    hip_amp: float = 0.30       # hip-pitch back->front swing amplitude over swing (rad)
    knee_amp: float = 0.55      # knee flexion (foot lift) at mid-swing (rad)
    ankle_frac: float = 0.5     # ankle counters knee by this fraction (keep the foot level)
    capture_gain: float = 1.4   # capture step: CoM-over-feet offset -> hip pitch (rad / m)
    capture_lead: float = 0.12  # CoM-velocity lead time in the capture point (s)
    forward_gain: float = 0.45  # v_cmd -> forward hip bias (rad per m/s)
    roll_amp: float = 0.0       # alternating frontal-plane weight shift (rad)
    roll_capture_gain: float = 0.0  # lateral CoM-over-feet feedback (rad / m)
    roll_capture_lead: float = 0.10 # lateral velocity lead time (s)
    roll_limit: float = 0.30         # total hip-roll reference bound (rad)
    hip_forward_sign: float = 1.0  # positive joint direction for forward swing; legacy default
    stance_sweep: bool = False     # continue front->back through stance, without a touchdown jump


class BipedalGait:
    """Alternating stepping gait -> per-joint target offsets (legs only; other joints untouched)."""

    def __init__(self, left_leg: Tuple[int, int, int], right_leg: Tuple[int, int, int],
                 n_joints: int, params: GaitParams = GaitParams(),
                 hip_roll: Tuple[int, int] | None = None):
        # left_leg / right_leg = (hip_pitch_idx, knee_idx, ankle_pitch_idx) in the joint vector.
        self._left = left_leg
        self._right = right_leg
        self._hip_roll = hip_roll
        self._n = int(n_joints)
        self.p = params

    def _leg_offsets(self, off, idx, phase, bias):
        hp, kn, an = idx
        p = self.p
        swing = (phase < p.swing_frac).astype(jnp.float32)         # 1 during swing, 0 in stance
        s = jnp.clip(phase / p.swing_frac, 0.0, 1.0)               # swing progress in [0, 1]
        lift = jnp.sin(jnp.pi * s) * swing                         # smooth 0 -> 1 -> 0 foot lift
        swing_hp = (2.0 * s - 1.0) * swing                         # hip swings back -> front
        if p.stance_sweep:
            stance_progress = jnp.clip(
                (phase - p.swing_frac) / (1.0 - p.swing_frac), 0.0, 1.0
            )
            swing_hp = swing_hp + (1.0 - 2.0 * stance_progress) * (1.0 - swing)
        hp_off = p.hip_forward_sign * (p.hip_amp * swing_hp + bias)
        kn_off = p.knee_amp * lift
        an_off = -p.ankle_frac * p.knee_amp * lift
        return off.at[hp].add(hp_off).at[kn].add(kn_off).at[an].add(an_off)

    def __call__(self, t, com_offset_x: float, com_vx: float, v_cmd: float,
                 com_offset_y: float = 0.0, com_vy: float = 0.0):
        """t: time (s); com_offset_x: CoM x minus feet-center x (>0 = leaning forward);
        com_vx: CoM forward velocity; v_cmd: forward base-velocity command. Returns (n_joints,)
        joint-target offsets (legs)."""
        p = self.p
        phase = (t * p.cadence) % 1.0
        capture = p.capture_gain * (com_offset_x + p.capture_lead * com_vx)
        bias = capture + p.forward_gain * v_cmd
        off = jnp.zeros((self._n,), jnp.float32)
        off = self._leg_offsets(off, self._left, phase, bias)
        off = self._leg_offsets(off, self._right, (phase + 0.5) % 1.0, bias)
        if self._hip_roll is not None:
            # At phase 0 the left leg enters swing, so positive roll shifts the
            # H1 pelvis toward the right stance foot; half a cycle later the
            # sign reverses for right-leg swing.  Applying the same sign to
            # both hips translates the pelvis laterally instead of merely
            # changing the foot spacing.
            roll = p.roll_amp * jnp.cos(2.0 * jnp.pi * phase)
            # On H1, positive equal hip-roll moves the pelvis toward -y.
            # Positive lateral CoM error therefore needs a positive correction.
            roll += p.roll_capture_gain * (
                com_offset_y + p.roll_capture_lead * com_vy
            )
            roll = jnp.clip(roll, -p.roll_limit, p.roll_limit)
            off = off.at[self._hip_roll[0]].add(roll)
            off = off.at[self._hip_roll[1]].add(roll)
        return off


__all__ = ["BipedalGait", "GaitParams"]
