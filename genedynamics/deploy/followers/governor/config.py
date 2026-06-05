"""Configuration for the reference governor (deploy-side, numpy runtime).

All margins / limits are explicit and config-driven so the module is reusable
across tasks and robots. Defaults are aligned with the G1 flat-locomotion
policy command envelope (see env.yaml ``limit_ranges``) and the corridor
follower's pelvis-roll gain.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GovernorConfig:
    """Knobs for the Step-2 reference-governor projection.

    The governor solves (per control step, in the corridor frame)::

        r_g = argmin_r  ||r - r*||_Q^2 + eta ||r - r_prev||^2
              s.t.  r in R_adm,  ||r - r_prev|| <= Delta_max

    where R_adm = {safety (tightened by m_track), rate (Delta_max), posture
    (pelvis-roll margin)}.
    """

    dt: float = 0.02  # control period (s); G1 policy runs at 50 Hz

    # --- Step-2 objective weights -------------------------------------------
    q_pos: float = 1.0   # tracking weight on (x, y)
    q_yaw: float = 1.0   # tracking weight on yaw
    eta: float = 0.0     # smoothness / inertia toward r_prev (0 => no smoothing)

    # --- Rate limits Delta_max (also the policy command envelope) -----------
    # These double as "reference feasibility": never command a velocity the
    # tracking policy was not trained to follow.
    v_max_lon: float = 1.0    # forward m/s   (env.yaml limit lin_vel_x max)
    v_min_lon: float = -0.5   # backward m/s  (lin_vel_x min)
    v_max_lat: float = 0.3    # lateral m/s   (lin_vel_y)
    yaw_rate_max: float = 0.2  # rad/s        (ang_vel_z)

    # --- Posture margin: phi_pelvis = -pelvis_roll_gain * v_lat -------------
    # Constraint |phi_pelvis| <= phi_max - m_phi  =>  bound on lateral step.
    pelvis_roll_gain: float = 0.08  # matches HumanoidTaskBuilderConfig
    phi_max: float = 0.30           # rad, pelvis-roll hard max
    m_phi: float = 0.05             # rad, posture stability margin

    # --- Safety tightening m_track (Step 4) ---------------------------------
    # Reserve margin so that, after tracking error eps_track, the real state
    # still satisfies g(x) <= 0. For an SDF (1-Lipschitz) m_track ~= eps_track.
    m_track: float = 0.05      # m; set by calibrate() = L_g * eps_track
    body_half_width: float = 0.0  # m; add if the safety SDF is point-based

    # --- Optional closed-loop error throttle (mode B) -----------------------
    closed_loop: bool = False      # False => feedforward governor (default)
    error_deadband: float = 0.05   # m; no throttle below this tracking error
    error_throttle_gain: float = 2.0  # 1/m; shrink Delta_max above deadband
