"""Unit tests for the generic ContactModel subsystem."""

from __future__ import annotations

import numpy as np
import pytest

from genedynamics.core.contact import (
    ContactModel,
    ContactState,
    RigidWithFrictionContact,
    SdfQuery,
    SpringDamperContact,
    force_bound_rows,
    penetration_bound_rows,
)


# ----- analytic SDF for a unit sphere centered at origin -------------------


class _SphereSdf:
    """Toy SDF satisfying the SdfQuery duck-type protocol.

    sdf(p) = ‖p‖ − r;  ∇sdf(p) = p / ‖p‖ (outward unit normal everywhere).
    """

    def __init__(self, radius: float = 0.5) -> None:
        self.radius = float(radius)

    def sdf_and_grad(self, points, *, backend: str = "numpy"):
        if backend == "jax":
            import jax.numpy as jnp
            p = jnp.asarray(points)
            r = jnp.linalg.norm(p, axis=-1)
            phi = r - self.radius
            grad = p / jnp.maximum(r[..., None], 1e-12)
            return phi, grad
        p = np.asarray(points, dtype=np.float32)
        r = np.linalg.norm(p, axis=-1)
        phi = (r - self.radius).astype(np.float32)
        grad = (p / np.maximum(r[..., None], 1e-12)).astype(np.float32)
        return phi, grad


def test_sphere_sdf_satisfies_protocol() -> None:
    assert isinstance(_SphereSdf(), SdfQuery)


# ----- spring-damper: kinematic state correctness --------------------------


def test_spring_damper_protocol_conformance() -> None:
    contact = SpringDamperContact(stiffness=100.0, damping=1.0)
    assert isinstance(contact, ContactModel)


def test_spring_damper_state_outside() -> None:
    """Tool well outside surface: in_contact=False, force=0."""
    sdf = _SphereSdf(radius=0.5)
    contact = SpringDamperContact(stiffness=100.0, damping=1.0)
    p = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v = np.zeros(3, dtype=np.float32)
    s = contact.query(p, v, sdf)
    assert bool(s.in_contact) is False
    np.testing.assert_allclose(np.asarray(s.force), 0.0, atol=1e-6)
    # Penetration is 0 outside contact.
    np.testing.assert_allclose(float(s.penetration), 0.0)


def test_spring_damper_state_inside_pure_normal() -> None:
    """Static tool penetrating along +x: force = k·d along +x."""
    sdf = _SphereSdf(radius=0.5)
    k = 100.0
    contact = SpringDamperContact(stiffness=k, damping=1.0)
    p = np.array([0.4, 0.0, 0.0], dtype=np.float32)   # phi = -0.1
    v = np.zeros(3, dtype=np.float32)
    s = contact.query(p, v, sdf)
    assert bool(s.in_contact) is True
    np.testing.assert_allclose(float(s.penetration), 0.1, atol=1e-6)
    # Surface point should land on the unit-normal sphere of radius 0.5.
    np.testing.assert_allclose(np.linalg.norm(np.asarray(s.surface_point)), 0.5, atol=1e-6)
    # Outward normal at p = (0.4, 0, 0) is (1, 0, 0).
    np.testing.assert_allclose(np.asarray(s.normal), [1.0, 0.0, 0.0], atol=1e-6)
    # Force = k * d * n = 100 * 0.1 * (1,0,0).
    np.testing.assert_allclose(np.asarray(s.force), [10.0, 0.0, 0.0], atol=1e-5)


def test_tangent_projector_idempotent_and_orthogonal_to_normal() -> None:
    """P_T = I - n n^T must be a rank-2 projector with P_T n = 0."""
    sdf = _SphereSdf(radius=0.5)
    contact = SpringDamperContact(stiffness=100.0, damping=1.0)
    p = np.array([0.6, 0.2, 0.1], dtype=np.float32)
    s = contact.query(p, np.zeros(3, dtype=np.float32), sdf)
    PT = np.asarray(s.tangent_projector)
    n = np.asarray(s.normal)
    np.testing.assert_allclose(PT @ PT, PT, atol=1e-5)
    np.testing.assert_allclose(PT @ n, 0.0, atol=1e-5)


def test_damping_term_pressing_in_increases_force() -> None:
    """ḋ = -∇φ · v > 0 when v points into surface; force grows accordingly."""
    sdf = _SphereSdf(radius=0.5)
    k, b = 100.0, 5.0
    contact = SpringDamperContact(stiffness=k, damping=b)
    p = np.array([0.45, 0.0, 0.0], dtype=np.float32)   # d=0.05
    v_in = np.array([-1.0, 0.0, 0.0], dtype=np.float32)  # pressing in
    s = contact.query(p, v_in, sdf)
    f_n = float(np.asarray(s.force)[0])
    expected = k * 0.05 + b * 1.0
    np.testing.assert_allclose(f_n, expected, atol=1e-3)


def test_spatially_varying_stiffness() -> None:
    """Callable stiffness gets the surface point as input."""
    def k_of(ps):
        # Stiff at the +x pole, soft at the -x pole.
        return 50.0 + 50.0 * ps[..., 0]

    sdf = _SphereSdf(radius=0.5)
    contact = SpringDamperContact(stiffness=k_of, damping=0.0)
    p_pos = np.array([0.45, 0.0, 0.0], dtype=np.float32)   # surface_x ≈ 0.5
    p_neg = np.array([-0.45, 0.0, 0.0], dtype=np.float32)  # surface_x ≈ -0.5
    v = np.zeros(3, dtype=np.float32)
    f_pos = float(np.linalg.norm(np.asarray(contact.query(p_pos, v, sdf).force)))
    f_neg = float(np.linalg.norm(np.asarray(contact.query(p_neg, v, sdf).force)))
    # At +x surface, stiffness ≈ 50 + 50·0.5 = 75; at -x, 50 - 25 = 25.
    np.testing.assert_allclose(f_pos / f_neg, 75.0 / 25.0, rtol=5e-2)


# ----- backend dispatch: jax matches numpy ---------------------------------


def test_jax_backend_matches_numpy() -> None:
    pytest.importorskip("jax")
    import jax.numpy as jnp
    sdf = _SphereSdf(radius=0.5)
    contact_np = SpringDamperContact(stiffness=100.0, damping=2.0, backend="numpy")
    contact_jx = SpringDamperContact(stiffness=100.0, damping=2.0, backend="jax")
    p = np.array([0.4, 0.1, -0.05], dtype=np.float32)
    v = np.array([-0.5, 0.2, 0.0], dtype=np.float32)
    s_np = contact_np.query(p, v, sdf)
    s_jx = contact_jx.query(jnp.asarray(p), jnp.asarray(v), sdf)
    for field in ("surface_point", "normal", "tangent_projector", "force"):
        np.testing.assert_allclose(
            np.asarray(getattr(s_jx, field)),
            np.asarray(getattr(s_np, field)),
            atol=1e-5,
        )
    np.testing.assert_allclose(float(s_jx.penetration), float(s_np.penetration), atol=1e-6)


def test_jax_query_under_jit() -> None:
    pytest.importorskip("jax")
    import jax, jax.numpy as jnp
    sdf = _SphereSdf(radius=0.5)
    contact = SpringDamperContact(stiffness=100.0, damping=2.0, backend="jax")
    p = jnp.array([0.4, 0.1, -0.05], dtype=jnp.float32)
    v = jnp.array([-0.5, 0.2, 0.0], dtype=jnp.float32)
    f_jit = jax.jit(lambda p, v: contact.query(p, v, sdf).force)
    out = f_jit(p, v)
    assert out.shape == (3,)


# ----- rigid contact -------------------------------------------------------


def test_rigid_friction_protocol_and_zero_force() -> None:
    contact = RigidWithFrictionContact(mu=0.4)
    assert isinstance(contact, ContactModel)
    sdf = _SphereSdf(radius=0.5)
    p = np.array([0.4, 0.0, 0.0], dtype=np.float32)
    s = contact.query(p, np.zeros(3, dtype=np.float32), sdf)
    np.testing.assert_allclose(np.asarray(s.force), 0.0, atol=1e-6)
    assert bool(s.in_contact) is True
    np.testing.assert_allclose(np.asarray(s.normal), [1.0, 0.0, 0.0], atol=1e-6)


# ----- CBF helpers ---------------------------------------------------------


def test_force_bound_rows_shape_and_signs() -> None:
    """Two-sided force bound emits exactly 2 rows in canonical Au ≤ b form."""
    sdf = _SphereSdf(radius=0.5)
    contact = SpringDamperContact(stiffness=100.0, damping=2.0)
    p = np.array([0.45, 0.0, 0.0], dtype=np.float32)
    v = np.zeros(3, dtype=np.float32)
    state = contact.query(p, v, sdf)
    n_dof = 4
    # A toy Jacobian: top 3 rows are an arbitrary 3×4 matrix.
    J = np.random.RandomState(0).randn(6, n_dof).astype(np.float32)
    A, b = force_bound_rows(state, J, stiffness=100.0, damping=2.0,
                            f_min=0.0, f_max=15.0)
    assert A.shape == (2, n_dof)
    assert b.shape == (2,)
    # The two rows are sign-flipped of each other (one for each bound).
    np.testing.assert_allclose(A[0], -A[1], atol=1e-6)


def test_penetration_bound_rows_two_sided() -> None:
    sdf = _SphereSdf(radius=0.5)
    contact = SpringDamperContact(stiffness=100.0, damping=2.0)
    p = np.array([0.4, 0.0, 0.0], dtype=np.float32)
    state = contact.query(p, np.zeros(3, dtype=np.float32), sdf)
    n_dof = 7
    J = np.random.RandomState(1).randn(6, n_dof).astype(np.float32)
    A, b = penetration_bound_rows(state, J, phi_target=-0.05,
                                  alpha=1.0, side="both")
    assert A.shape == (2, n_dof)
    assert b.shape == (2,)


def test_penetration_bound_rows_one_sided() -> None:
    sdf = _SphereSdf(radius=0.5)
    contact = SpringDamperContact(stiffness=100.0, damping=2.0)
    state = contact.query(
        np.array([0.4, 0.0, 0.0], dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        sdf,
    )
    J = np.random.RandomState(2).randn(6, 5).astype(np.float32)
    A, b = penetration_bound_rows(state, J, side="upper")
    assert A.shape == (1, 5) or A.shape == (5,)  # accepting either
