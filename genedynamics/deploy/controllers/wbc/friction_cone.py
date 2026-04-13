"""Inequality constraints for the WBC QP.

This module assembles all of the ``G · x ≤ h`` rows of the WBC QP, where
``x = [ddq; λ]``. Five families of inequalities are produced, in this
order:

1. **Friction pyramid** on every support contact:
   ``|fx| ≤ μ·fz`` and ``|fy| ≤ μ·fz`` (4 rows per contact).
2. **Normal force bounds**:
   ``λ_min ≤ fz ≤ λ_max`` (2 rows per contact).
3. **Tangent and moment magnitude bounds**:
   ``|fx|, |fy| ≤ λ_max_tangent`` (4 rows) and per-axis moment bounds
   ``|mx| ≤ Mx``, ``|my| ≤ My``, ``|mz| ≤ Mz`` (6 rows per contact).
4. **Actuator torque bounds**:
   ``|S · (M·ddq + bias − Jcᵀ·λ)| ≤ τ_max`` (2 rows per actuator with a
   finite torque limit).
5. **Acceleration bounds**:
   ``|ddq_i| ≤ max_base_accel`` for the floating base, ``≤ max_joint_accel``
   for the rest (2 rows per ``nv`` index).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from genedynamics.deploy.controllers.wbc.config import LimitsConfig
from genedynamics.deploy.controllers.wbc.contact_blocks import SupportContactBlock
from genedynamics.robots.g1 import G1RobotSpec

__all__ = ["build_inequality_rows"]


def build_inequality_rows(
    spec: G1RobotSpec,
    M: np.ndarray,
    bias: np.ndarray,
    blocks: List[SupportContactBlock],
    *,
    nv: int,
    limits: LimitsConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assemble the full inequality matrix for the WBC QP.

    Returns ``(G, h)`` such that ``G · [ddq; λ] ≤ h`` enforces friction,
    normal, tangent, moment, torque and acceleration bounds.
    """
    n = len(blocks)
    n_lambda = 6 * n
    total_dim = nv + n_lambda
    rows: List[np.ndarray] = []
    rhs: List[float] = []

    # ---- 1–3. Friction cone + magnitude bounds (per contact) -------------
    if n_lambda > 0:
        mu = float(limits.friction_coeff)
        for i in range(n):
            base = nv + 6 * i
            # 1. Friction pyramid: |fx| ≤ μ·fz, |fy| ≤ μ·fz
            for ax, sign in ((0, +1), (0, -1), (1, +1), (1, -1)):
                row = np.zeros(total_dim, dtype=np.float64)
                row[base + ax] = float(sign)
                row[base + 2] = -mu
                rows.append(row)
                rhs.append(0.0)

            # 2. Normal-force bounds: −fz ≤ −λ_min, fz ≤ λ_max
            row = np.zeros(total_dim, dtype=np.float64)
            row[base + 2] = -1.0
            rows.append(row)
            rhs.append(-float(limits.lambda_min_normal))

            row = np.zeros(total_dim, dtype=np.float64)
            row[base + 2] = 1.0
            rows.append(row)
            rhs.append(float(limits.lambda_max_normal))

            # 3a. Tangent magnitude bounds (4 rows)
            for ax, sign in ((0, +1), (0, -1), (1, +1), (1, -1)):
                row = np.zeros(total_dim, dtype=np.float64)
                row[base + ax] = float(sign)
                rows.append(row)
                rhs.append(float(limits.lambda_max_tangent))

            # 3b. Per-axis moment bounds (6 rows)
            moment_bounds = (
                limits.lambda_max_moment_roll,
                limits.lambda_max_moment_pitch,
                limits.lambda_max_moment_yaw,
            )
            for axis, bound in enumerate(moment_bounds):
                for sign in (+1, -1):
                    row = np.zeros(total_dim, dtype=np.float64)
                    row[base + 3 + axis] = float(sign)
                    rows.append(row)
                    rhs.append(float(bound))

    # ---- 4. Actuator torque bounds (only for finite limits) --------------
    act_idx = spec.actuated_dof_indices
    num_act = spec.num_actuated
    tau_map = np.zeros((num_act, total_dim), dtype=np.float64)
    tau_map[:, :nv] = M[act_idx, :]
    if n_lambda > 0 and blocks:
        support_jacobian = np.vstack([block.J for block in blocks])
        tau_map[:, nv:] = -support_jacobian[:, act_idx].T
    tau_bias = np.asarray(bias[act_idx], dtype=np.float64)
    torque_limit = limits.torque_limit_scale * spec.torque_limit_vector()
    finite_mask = np.isfinite(torque_limit)

    for i, finite in enumerate(finite_mask):
        if not finite:
            continue
        # +tau ≤ τ_max  →  +(S·M·ddq − S·Jᵀ·λ) ≤ τ_max − S·bias
        rows.append(tau_map[i])
        rhs.append(float(torque_limit[i] - tau_bias[i]))
        # −tau ≤ τ_max  →  −(S·M·ddq − S·Jᵀ·λ) ≤ τ_max + S·bias
        rows.append(-tau_map[i])
        rhs.append(float(torque_limit[i] + tau_bias[i]))

    # ---- 5. Acceleration bounds ------------------------------------------
    for i in range(nv):
        bound = float(limits.max_base_accel if i < 6 else limits.max_joint_accel)
        row = np.zeros(total_dim, dtype=np.float64)
        row[i] = 1.0
        rows.append(row)
        rhs.append(bound)
        row = np.zeros(total_dim, dtype=np.float64)
        row[i] = -1.0
        rows.append(row)
        rhs.append(bound)

    if not rows:
        return (
            np.zeros((0, total_dim), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )
    return np.vstack(rows), np.asarray(rhs, dtype=np.float64)
