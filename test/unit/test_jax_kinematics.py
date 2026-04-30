"""Unit tests for JAXKinematics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("urchin")

from genedynamics.envs.robots.jax_kinematics import JAXKinematics
from genedynamics.envs.robots.kinematics_protocol import KinematicsProtocol


URDF_DIR = Path(__file__).resolve().parents[1] / "data" / "urdf"
PLANAR2 = URDF_DIR / "2dof_planar.urdf"
PRISMATIC_REVOLUTE = URDF_DIR / "prismatic_revolute.urdf"
PANDA = (
    Path(__file__).resolve().parents[2]
    / "third_party/environments/d3il/models/common/robots/panda_arm_hand_pinocchio.urdf"
)


# ---------------------------------------------------------------------------
# 2-DoF planar arm: hand-computable ground truth
# ---------------------------------------------------------------------------


def _planar_kin() -> JAXKinematics:
    return JAXKinematics.from_urdf(PLANAR2, ee_link="ee")


def test_protocol_conformance() -> None:
    kin = _planar_kin()
    assert isinstance(kin, KinematicsProtocol)


def test_n_dof_and_names_planar() -> None:
    kin = _planar_kin()
    assert kin.n_dof == 2
    assert kin.joint_names == ("joint1", "joint2")


@pytest.mark.parametrize(
    "q",
    [
        np.zeros(2, dtype=np.float32),
        np.array([0.5, 0.5], dtype=np.float32),
        np.array([0.7, -0.3], dtype=np.float32),
        np.array([np.pi / 3, np.pi / 4], dtype=np.float32),
    ],
)
def test_fk_planar_against_analytic(q: np.ndarray) -> None:
    """Closed-form FK: p_ee = (cos q1 + cos(q1+q2), sin q1 + sin(q1+q2), 0)."""
    kin = _planar_kin()
    T = np.asarray(kin.fk(jnp.asarray(q)))
    expected = np.array([
        np.cos(q[0]) + np.cos(q[0] + q[1]),
        np.sin(q[0]) + np.sin(q[0] + q[1]),
        0.0,
    ])
    np.testing.assert_allclose(T[:3, 3], expected, atol=1e-5)
    # Rotation should be a rotation about Z by (q1 + q2).
    theta = q[0] + q[1]
    expected_R = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    np.testing.assert_allclose(T[:3, :3], expected_R, atol=1e-5)


def test_jacobian_top_matches_autodiff_planar() -> None:
    """Top 3 rows of geometric J equal the analytic Jacobian of pos_ee(q)."""
    kin = _planar_kin()
    q = jnp.array([0.4, -0.7], dtype=jnp.float32)
    J = np.asarray(kin.jacobian(q))
    pos = lambda q: kin.fk(q)[:3, 3]
    J_pos = np.asarray(jax.jacfwd(pos)(q))
    np.testing.assert_allclose(J[:3, :], J_pos, atol=1e-5)


def test_jacobian_angular_planar() -> None:
    """Both joints rotate about +Z; bottom 3 rows must be (0,0,1) per column."""
    kin = _planar_kin()
    q = jnp.array([0.4, -0.7], dtype=jnp.float32)
    J = np.asarray(kin.jacobian(q))
    np.testing.assert_allclose(J[3:, 0], [0.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(J[3:, 1], [0.0, 0.0, 1.0], atol=1e-6)


def test_jit_fk_matches_eager() -> None:
    kin = _planar_kin()
    q = jnp.array([0.3, 0.6], dtype=jnp.float32)
    fk_jit = jax.jit(kin.fk)
    np.testing.assert_allclose(np.asarray(fk_jit(q)), np.asarray(kin.fk(q)), atol=1e-6)


def test_vmap_fk_batch_shape() -> None:
    kin = _planar_kin()
    q_batch = jnp.array(np.random.RandomState(0).uniform(-1, 1, (8, 2)), dtype=jnp.float32)
    Ts = jax.vmap(kin.fk)(q_batch)
    assert Ts.shape == (8, 4, 4)


def test_grad_via_finite_difference_planar() -> None:
    kin = _planar_kin()
    q = jnp.array([0.4, -0.7], dtype=jnp.float32)
    pos = lambda q: kin.fk(q)[:3, 3]
    J_ad = np.asarray(jax.jacfwd(pos)(q))
    eps = 1e-3
    J_fd = np.zeros_like(J_ad)
    for i in range(2):
        dq = np.zeros(2, dtype=np.float32); dq[i] = eps
        p_plus = np.asarray(pos(q + dq))
        p_minus = np.asarray(pos(q - dq))
        J_fd[:, i] = (p_plus - p_minus) / (2 * eps)
    np.testing.assert_allclose(J_ad, J_fd, atol=1e-3)


# ---------------------------------------------------------------------------
# Prismatic + revolute — exercise the prismatic branch
# ---------------------------------------------------------------------------


def test_prismatic_branch_fk() -> None:
    kin = JAXKinematics.from_urdf(PRISMATIC_REVOLUTE, ee_link="ee")
    assert kin.n_dof == 2
    # See URDF docstring: p_ee = (1 + cos theta, sin theta, 0.5 + d)
    q = jnp.array([0.3, np.pi / 6], dtype=jnp.float32)
    T = np.asarray(kin.fk(q))
    expected = np.array([1.0 + np.cos(np.pi / 6), np.sin(np.pi / 6), 0.3 + 0.5])
    np.testing.assert_allclose(T[:3, 3], expected, atol=1e-5)


def test_prismatic_jacobian_column_zero_angular() -> None:
    """Prismatic joint contributes pure linear velocity along its axis."""
    kin = JAXKinematics.from_urdf(PRISMATIC_REVOLUTE, ee_link="ee")
    q = jnp.array([0.2, 0.5], dtype=jnp.float32)
    J = np.asarray(kin.jacobian(q))
    # Column 0 = lift_joint (prismatic +Z): linear (0,0,1), angular (0,0,0).
    np.testing.assert_allclose(J[:3, 0], [0.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(J[3:, 0], [0.0, 0.0, 0.0], atol=1e-6)


# ---------------------------------------------------------------------------
# Damped pinv & nullspace
# ---------------------------------------------------------------------------


def test_pinv_left_inverse_planar() -> None:
    """For full column rank J (n=2 ≤ m=6), J† J = I."""
    kin = _planar_kin()
    q = jnp.array([0.7, 0.4], dtype=jnp.float32)
    J = np.asarray(kin.jacobian(q))
    Jp = np.asarray(kin.jacobian_pinv(q, damping=1e-4))
    np.testing.assert_allclose(Jp @ J, np.eye(2), atol=1e-3)


def test_nullspace_annihilates_jacobian_panda() -> None:
    """Away from singularity, N @ J^T should be ≈ 0."""
    if not PANDA.exists():
        pytest.skip(f"panda URDF not available at {PANDA}")
    kin = JAXKinematics.from_urdf(PANDA, ee_link="panda_link8")
    # A non-singular config (not q=0).
    q = jnp.array([0.3, -0.4, 0.5, -1.5, 0.6, 1.2, 0.8], dtype=jnp.float32)
    N = np.asarray(kin.nullspace(q))
    J = np.asarray(kin.jacobian(q))
    np.testing.assert_allclose(N @ J.T, 0.0, atol=1e-4)
    # And N is idempotent (away from singularity).
    np.testing.assert_allclose(N @ N, N, atol=1e-4)


# ---------------------------------------------------------------------------
# Real 7-DoF Panda (skipped if URDF missing)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not PANDA.exists(), reason="panda URDF not available")
def test_panda_loads_seven_dof() -> None:
    kin = JAXKinematics.from_urdf(PANDA, ee_link="panda_link8")
    assert kin.n_dof == 7
    lo, hi = kin.joint_limits
    assert lo.shape == (7,)
    assert hi.shape == (7,)
    assert np.all(np.asarray(lo) < np.asarray(hi))


@pytest.mark.skipif(not PANDA.exists(), reason="panda URDF not available")
def test_panda_jit_vmap() -> None:
    kin = JAXKinematics.from_urdf(PANDA, ee_link="panda_link8")
    q_batch = jnp.array(
        np.random.RandomState(0).uniform(-0.5, 0.5, (16, 7)), dtype=jnp.float32
    )
    fk_v = jax.jit(jax.vmap(kin.fk))
    Ts = fk_v(q_batch)
    assert Ts.shape == (16, 4, 4)
    # Last column (3,3) entry of every transform must be 1 (homogeneous).
    np.testing.assert_allclose(np.asarray(Ts[:, 3, 3]), 1.0, atol=1e-6)
