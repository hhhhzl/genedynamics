"""Stage 7 — Carry / transport reward (loco-manipulation).

CPU, no MPM rollout: unit-tests the `carry_reward` pure function on synthetic
object trajectories. The contact-stability term G_T (fraction of time the object
stays carried) and the drop penalty are the carry-specific pieces; the full
rollout (rollout_return_carry, gravity-loaded manipuland) is GPU-verified.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from genedynamics.envs.external.jax_mpm.scene import carry_reward  # noqa: E402


def _traj(x0, x1, y_series, T=50, z=0.5):
    xs = np.linspace(x0, x1, T)
    ys = np.full(T, y_series) if np.isscalar(y_series) else np.asarray(y_series)
    zs = np.full(T, z)
    return jnp.asarray(np.stack([xs, ys, zs], axis=1).astype(np.float32))


INIT = jnp.asarray([0.5, 0.12, 0.5], dtype=jnp.float32)


def test_carried_beats_dropped():
    carried = _traj(0.5, 0.8, 0.12)                          # forward + stays up
    T = 50
    dropped = _traj(0.5, 0.55, np.linspace(0.12, 0.05, T))   # sags to floor
    r_c, _, g_c = carry_reward(carried, INIT, goal_x=1.0)
    r_d, _, g_d = carry_reward(dropped, INIT, goal_x=1.0)
    assert float(r_c) > float(r_d)
    assert float(g_c) > float(g_d)


def test_carried_frac_in_unit_range():
    tr = _traj(0.5, 0.7, 0.12)
    _, _, g = carry_reward(tr, INIT, goal_x=1.0)
    assert 0.0 <= float(g) <= 1.0


def test_forward_transport_increases_reward():
    r_near, _, _ = carry_reward(_traj(0.5, 0.55, 0.12), INIT, goal_x=1.0)
    r_far, _, _ = carry_reward(_traj(0.5, 0.90, 0.12), INIT, goal_x=1.0)
    assert float(r_far) > float(r_near)


def test_drop_penalty_at_final_floor():
    T = 30
    ys_drop = np.concatenate([np.full(T - 1, 0.12), [0.05]])  # ends on floor
    r_drop, _, _ = carry_reward(_traj(0.5, 0.7, ys_drop, T=T), INIT, goal_x=1.0)
    r_ok, _, _ = carry_reward(_traj(0.5, 0.7, 0.12, T=T), INIT, goal_x=1.0)
    assert float(r_ok) > float(r_drop)
