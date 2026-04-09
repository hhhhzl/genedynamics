"""Tests for the friction-cone / torque-bound inequality builder.

The builder produces the ``G x ≤ h`` rows for ``x = [ddq; λ]``. We hit it
with synthetic spec stand-ins so the math is fully testable without
MuJoCo.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest

from genedynamics.deploy.controllers.wbc.config import LimitsConfig
from genedynamics.deploy.controllers.wbc.contact_blocks import SupportContactBlock
from genedynamics.deploy.controllers.wbc.friction_cone import build_inequality_rows


# ---------------------------------------------------------------------------
# Synthetic spec
# ---------------------------------------------------------------------------


@dataclass
class _DummySpec:
    """Minimal RobotSpec stand-in: 3 actuated DoFs starting at index 6."""

    num_actuated: int = 3
    actuated_joints: tuple = ("a", "b", "c")
    _torque_limits: np.ndarray = field(
        default_factory=lambda: np.array([10.0, 20.0, np.inf])
    )

    @property
    def actuated_dof_indices(self) -> np.ndarray:
        return np.array([6, 7, 8], dtype=np.int32)

    def torque_limit_vector(self) -> np.ndarray:
        return self._torque_limits.copy()


def _empty_M_bias(nv: int):
    return np.zeros((nv, nv)), np.zeros(nv)


# ---------------------------------------------------------------------------
# Friction pyramid: row count
# ---------------------------------------------------------------------------


def test_friction_cone_row_count_no_contacts():
    spec = _DummySpec()
    nv = 9
    M, bias = _empty_M_bias(nv)
    G, h = build_inequality_rows(spec, M, bias, blocks=[], nv=nv, limits=LimitsConfig())
    # 0 contacts → 0 friction rows
    # + 4 finite torque rows (2 actuators × 2 signs)
    # + 18 acceleration rows (9 nv × 2 signs)
    expected = 0 + 4 + 2 * nv
    assert G.shape == (expected, nv)
    assert h.shape == (expected,)


def test_friction_cone_row_count_double_support():
    spec = _DummySpec()
    nv = 9
    M, bias = _empty_M_bias(nv)
    block = SupportContactBlock(
        name="left_foot",
        J=np.zeros((6, nv)),
        a_des=np.zeros(6),
    )
    blocks = [block, SupportContactBlock(name="right_foot", J=np.zeros((6, nv)), a_des=np.zeros(6))]
    G, h = build_inequality_rows(spec, M, bias, blocks=blocks, nv=nv, limits=LimitsConfig())

    # Per contact:
    #   4 friction pyramid + 2 normal bounds + 4 tangent bounds + 6 moment bounds
    #   = 16
    # Plus the same torque (4 rows) + acceleration (18 rows) sections.
    per_contact = 4 + 2 + 4 + 6
    expected = 2 * per_contact + 4 + 2 * nv
    n_lambda = 6 * 2
    assert G.shape == (expected, nv + n_lambda)
    assert h.shape == (expected,)


# ---------------------------------------------------------------------------
# Friction pyramid: bound semantics
# ---------------------------------------------------------------------------


def test_friction_pyramid_uses_configured_mu():
    spec = _DummySpec()
    nv = 9
    block = SupportContactBlock(name="left_foot", J=np.zeros((6, nv)), a_des=np.zeros(6))
    limits = LimitsConfig(friction_coeff=0.7)
    G, h = build_inequality_rows(spec, _empty_M_bias(nv)[0], _empty_M_bias(nv)[1],
                                  blocks=[block], nv=nv, limits=limits)

    # First four rows are the pyramid: ±fx − μ·fz ≤ 0 and ±fy − μ·fz ≤ 0
    fx_pos = G[0]
    assert fx_pos[nv + 0] == 1.0
    assert fx_pos[nv + 2] == -0.7
    assert h[0] == 0.0

    fx_neg = G[1]
    assert fx_neg[nv + 0] == -1.0
    assert fx_neg[nv + 2] == -0.7

    fy_pos = G[2]
    assert fy_pos[nv + 1] == 1.0
    assert fy_pos[nv + 2] == -0.7


def test_normal_force_bounds_use_configured_min_and_max():
    spec = _DummySpec()
    nv = 9
    block = SupportContactBlock(name="left_foot", J=np.zeros((6, nv)), a_des=np.zeros(6))
    limits = LimitsConfig(lambda_min_normal=15.0, lambda_max_normal=600.0)
    G, h = build_inequality_rows(spec, _empty_M_bias(nv)[0], _empty_M_bias(nv)[1],
                                  blocks=[block], nv=nv, limits=limits)

    # Rows 4 and 5 are normal force bounds (after the 4 pyramid rows).
    assert G[4, nv + 2] == -1.0 and h[4] == -15.0
    assert G[5, nv + 2] == 1.0 and h[5] == 600.0


def test_tangent_magnitude_bounds():
    spec = _DummySpec()
    nv = 9
    block = SupportContactBlock(name="left_foot", J=np.zeros((6, nv)), a_des=np.zeros(6))
    limits = LimitsConfig(lambda_max_tangent=180.0)
    G, h = build_inequality_rows(spec, _empty_M_bias(nv)[0], _empty_M_bias(nv)[1],
                                  blocks=[block], nv=nv, limits=limits)

    # Rows 6..9 are the tangent magnitude bounds.
    for r in (6, 7, 8, 9):
        assert h[r] == 180.0


def test_per_axis_moment_bounds():
    spec = _DummySpec()
    nv = 9
    block = SupportContactBlock(name="left_foot", J=np.zeros((6, nv)), a_des=np.zeros(6))
    limits = LimitsConfig(
        lambda_max_moment_roll=50.0,
        lambda_max_moment_pitch=60.0,
        lambda_max_moment_yaw=40.0,
    )
    G, h = build_inequality_rows(spec, _empty_M_bias(nv)[0], _empty_M_bias(nv)[1],
                                  blocks=[block], nv=nv, limits=limits)

    # Rows 10..15 are six moment-axis bounds (axis × ±sign).
    assert h[10] == 50.0 and h[11] == 50.0
    assert h[12] == 60.0 and h[13] == 60.0
    assert h[14] == 40.0 and h[15] == 40.0


# ---------------------------------------------------------------------------
# Torque bounds: only finite limits produce rows
# ---------------------------------------------------------------------------


def test_torque_bounds_skip_infinite_limits():
    spec = _DummySpec()  # third actuator has inf torque limit
    nv = 9
    M, bias = _empty_M_bias(nv)
    G, h = build_inequality_rows(spec, M, bias, blocks=[], nv=nv, limits=LimitsConfig())

    # 2 actuators with finite limits × 2 signs = 4 torque rows
    accel_rows = 2 * nv
    torque_rows = G.shape[0] - accel_rows
    assert torque_rows == 4


def test_torque_bounds_use_torque_limit_scale():
    """Halving the torque scale should halve the torque-row RHS magnitudes."""
    spec = _DummySpec()
    nv = 9
    M = np.eye(nv)  # M[6, :] gives identity row for first actuator
    bias = np.zeros(nv)

    full = build_inequality_rows(spec, M, bias, blocks=[], nv=nv,
                                  limits=LimitsConfig(torque_limit_scale=1.0))
    half = build_inequality_rows(spec, M, bias, blocks=[], nv=nv,
                                  limits=LimitsConfig(torque_limit_scale=0.5))

    # Torque rows are first 4 rows (no contacts), each ± per actuator.
    # First two rows are actuator 0 (limit 10) → ±10.0 vs ±5.0.
    full_h, half_h = full[1], half[1]
    assert abs(full_h[0]) == pytest.approx(10.0)
    assert abs(half_h[0]) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Acceleration bounds
# ---------------------------------------------------------------------------


def test_acceleration_bounds_split_base_vs_joint():
    spec = _DummySpec()
    nv = 9  # first 6 = floating base, last 3 = actuated joints
    G, h = build_inequality_rows(
        spec,
        np.zeros((nv, nv)),
        np.zeros(nv),
        blocks=[],
        nv=nv,
        limits=LimitsConfig(max_base_accel=20.0, max_joint_accel=40.0),
    )

    # The last 2*nv rows are acceleration bounds.
    accel = G[-(2 * nv) :]
    accel_h = h[-(2 * nv) :]

    # First 12 are the floating-base 6 DoFs × 2 signs → bound 20.
    for i in range(12):
        assert accel_h[i] == 20.0
    # Remaining 6 are the joint DoFs × 2 signs → bound 40.
    for i in range(12, 18):
        assert accel_h[i] == 40.0
