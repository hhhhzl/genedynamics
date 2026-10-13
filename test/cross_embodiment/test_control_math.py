import numpy as np
import jax
import jax.numpy as jp

from genedynamics.learning.cross_embodiment.core.control_math import integrate_rotation


def test_zero_rotation_is_finite_in_forward_and_reverse_mode():
    fn = lambda velocity: integrate_rotation(jp, jp.eye(3), velocity, .02)
    jacobian = jax.jacfwd(fn)(jp.zeros(3))
    gradient = jax.grad(lambda v: jp.sum(fn(v) * jp.arange(9).reshape(3,3)))(jp.zeros(3))
    assert np.isfinite(jacobian).all()
    assert np.isfinite(gradient).all()
    np.testing.assert_array_equal(fn(jp.zeros(3)), np.eye(3))
    np.testing.assert_allclose(np.asarray(jacobian)[1,2,0], -.02, atol=1e-7)


def test_rotation_integration_preserves_so3():
    rotation = np.eye(3)
    for _ in range(500):
        rotation = integrate_rotation(np, rotation, np.array([.2,-.1,.3]), .02)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rotation), 1., atol=1e-12)


def test_rotation_obeys_input_dtype_when_another_solver_enabled_x64():
    with jax.experimental.enable_x64():
        result = integrate_rotation(jp, jp.eye(3, dtype=jp.float32), jp.zeros(3, dtype=jp.float32), .02)
        assert result.dtype == jp.float32
