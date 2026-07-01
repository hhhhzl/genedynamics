"""Phase-driven bipedal stepping gait — the leg-impedance balance substrate for a humanoid
manipulation pi_low (whole-body impedance = hand Cartesian impedance + leg joint impedance).

Why a gait at all: a STATIC stance cannot balance a humanoid push. The push (and the arm
reaching forward) shifts the CoM ahead of the feet; the ankle torque saturates (~40 Nm vs a
~50 Nm tipping moment) and the robot tips. The fix is to STEP — reposition the swing foot under
a CoM *capture point* so the support follows the CoM (dynamic balance). DIAL avoids the issue by
walking; we provide the equivalent as an explicit, reusable leg controller.

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


class BipedalGait:
    """Alternating stepping gait -> per-joint target offsets (legs only; other joints untouched)."""

    def __init__(self, left_leg: Tuple[int, int, int], right_leg: Tuple[int, int, int],
                 n_joints: int, params: GaitParams = GaitParams()):
        # left_leg / right_leg = (hip_pitch_idx, knee_idx, ankle_pitch_idx) in the joint vector.
        self._left = left_leg
        self._right = right_leg
        self._n = int(n_joints)
        self.p = params

    def _leg_offsets(self, off, idx, phase, bias):
        hp, kn, an = idx
        p = self.p
        swing = (phase < p.swing_frac).astype(jnp.float32)         # 1 during swing, 0 in stance
        s = jnp.clip(phase / p.swing_frac, 0.0, 1.0)               # swing progress in [0, 1]
        lift = jnp.sin(jnp.pi * s) * swing                         # smooth 0 -> 1 -> 0 foot lift
        swing_hp = (2.0 * s - 1.0) * swing                         # hip swings back -> front
        hp_off = p.hip_amp * swing_hp + bias                       # + capture/forward bias
        kn_off = p.knee_amp * lift
        an_off = -p.ankle_frac * p.knee_amp * lift
        return off.at[hp].add(hp_off).at[kn].add(kn_off).at[an].add(an_off)

    def __call__(self, t, com_offset_x: float, com_vx: float, v_cmd: float):
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
        return off


__all__ = ["BipedalGait", "GaitParams"]
