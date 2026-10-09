"""Numerical contract tests for the vendored PegasusFlow JAX reproduction."""

import jax.numpy as jnp
import numpy as np
import pytest

from genedynamics.solvers.single.pegasusflow.backends.pegasusflow_jax import (
    _catmull_rom_basis,
    _noise_decay,
    _noise_shape,
    _weighted_basis_update,
)


def test_catmull_rom_basis_matches_uniform_reference():
    # Numerical provenance: expand the cubic Hermite interpolant with centered
    # tangents (P[i+1] - P[i-1]) / 2 and duplicated boundary control points.
    # For five uniform knots and four samples per segment, the exact basis
    # coefficients are integer multiples of 1/128. This independent golden
    # reference needs neither the implementation nor an external checkout.
    expected = np.array([
        [128, 0, 0, 0, 0],
        [102, 29, -3, 0, 0],
        [64, 72, -8, 0, 0],
        [26, 111, -9, 0, 0],
        [0, 128, 0, 0, 0],
        [-9, 111, 29, -3, 0],
        [-8, 72, 72, -8, 0],
        [-3, 29, 111, -9, 0],
        [0, 0, 128, 0, 0],
        [0, -9, 111, 29, -3],
        [0, -8, 72, 72, -8],
        [0, -3, 29, 111, -9],
        [0, 0, 0, 128, 0],
        [0, 0, -9, 111, 26],
        [0, 0, -8, 72, 64],
        [0, 0, -3, 29, 102],
        [0, 0, 0, 0, 128],
    ], dtype=np.float32) / 128.0
    actual = _catmull_rom_basis(17, 5)
    assert actual.shape == (17, 5)
    np.testing.assert_allclose(
        actual, expected,
        rtol=0.0, atol=1e-7,
    )
    # Every knot, including both endpoints, is interpolated exactly; constant
    # control sequences remain constant between knots (partition of unity).
    np.testing.assert_allclose(actual[::4], np.eye(5), rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(actual.sum(axis=1), 1.0, rtol=0.0, atol=1e-7)


@pytest.mark.parametrize("method,gamma", [("wbfo", 0.0), ("avwbfo", 1.0), ("avwbfo", 0.9)])
def test_weighted_basis_update_matches_vendored_torch_equations(method, gamma):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(7)
    samples_np = rng.normal(size=(9, 5, 3)).astype(np.float32)
    rewards_np = rng.normal(size=(9, 17)).astype(np.float32)
    phi_np = _catmull_rom_basis(17, 5)

    samples = torch.tensor(samples_np)
    W = torch.tensor(rewards_np) @ torch.tensor(phi_np)
    if method == "avwbfo":
        discounted = torch.zeros_like(W)
        discounted[:, -1] = W[:, -1]
        for k in range(W.shape[1] - 2, -1, -1):
            discounted[:, k] = W[:, k] + gamma * discounted[:, k + 1]
        W = discounted
    # Upstream torch.std uses correction=1 (sample standard deviation).
    W = (W - W.mean(dim=0, keepdim=True)) / (W.std(dim=0, keepdim=True) + 1e-8)
    W = torch.softmax(W / 0.1, dim=0)
    expected = torch.einsum("sn,sna->na", W, samples).numpy()

    actual = np.asarray(_weighted_basis_update(
        jnp.asarray(samples_np), jnp.asarray(rewards_np), jnp.asarray(phi_np),
        temp=0.1, update_method=method, gamma=gamma,
    ))
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


def test_upstream_default_s2_linear_exponential_schedule():
    np.testing.assert_allclose(
        np.asarray(_noise_shape("linear", 5)),
        np.linspace(0.0, 1.0, 5, dtype=np.float32),
    )
    scales = [
        float(_noise_decay("exponential", k, 4, decay_rate=0.9, final_ratio=0.1))
        for k in range(4)
    ]
    np.testing.assert_allclose(scales, [1.0, 0.9, 0.81, 0.729], rtol=1e-6)
    assert float(_noise_decay(
        "exponential", 0, 1, decay_rate=0.9, final_ratio=0.1,
    )) == pytest.approx(1.0)
