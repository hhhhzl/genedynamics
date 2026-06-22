"""DIAL reverse-update + bridge composition tests on synthetic rewards.

Runs on fedguide x86 (no mjx): the dynamics rollout is a mock with a known
optimum, so we can verify the full DIAL algorithm (node sampling, self-normalised
MPPI weighting, geometric anneal, warm-started receding-horizon execution)
converges, and that ``RecedingHorizonController(DialReversePlanner)`` behaves
like DIAL-MPC. Only the real env rollout is deferred to arm64/GPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from genedynamics.solvers.single.dial.backends.dial_jax import (
    DialBackendJax,
    make_sigma_control,
    make_traj_diffuse_factors,
)
from genedynamics.solvers.common.receding_horizon import RecedingHorizonController

NU = 2
TARGET = 0.5  # mock reward optimum (inside the [-1,1] action box)


def _make_rollout(target=TARGET):
    # reward per step = -||u_h - target||^2 ; ignores state/time. us:(N,H,nu)->(N,H)
    def rollout_fn(state, us, t0=0.0):
        return -jnp.sum((us - target) ** 2, axis=-1)
    return rollout_fn


def _planner(Nsample=256, **kw):
    return DialBackendJax(
        rollout_fn=_make_rollout(), nu=NU, Hnode=4, Hsample=16, Nsample=Nsample,
        temp_sample=0.1, **kw,
    )


def test_schedule_shapes_and_anneal():
    sc = make_sigma_control(Hnode=4, horizon_diffuse_factor=0.9, sigma_scale=1.0)
    assert sc.shape == (5,)
    # later nodes get MORE noise (arange reversed)
    assert sc[-1] > sc[0]
    fac = make_traj_diffuse_factors(sc, traj_diffuse_factor=0.5, n_diffuse=3)
    assert fac.shape == (3, 5)
    # noise shrinks across reverse steps
    assert jnp.all(fac[1] < fac[0]) and jnp.all(fac[2] < fac[1])


def test_reverse_once_improves_over_incumbent():
    # One reverse step from zeros should move the weighted mean toward TARGET.
    p = _planner(Nsample=512)
    Y0 = p.init_plan_var()
    sched = p.make_schedule(1)
    Y1, info = p._reverse(None, jax.random.PRNGKey(0), Y0, sched[0])
    # dense control mean moved from 0 toward TARGET
    u0_mean = jnp.mean(p.spline.node2u(Y0))
    u1_mean = jnp.mean(p.spline.node2u(Y1))
    assert abs(float(u1_mean) - TARGET) < abs(float(u0_mean) - TARGET)
    # node-0 stays pinned to the incumbent (here zeros)
    assert jnp.allclose(Y1[0], Y0[0])


def test_replan_converges_toward_optimum():
    # Many warm-up diffusion steps from cold start should approach TARGET.
    p = _planner(Nsample=512)
    Y = p.init_plan_var()
    sched = p.make_schedule(40)
    Y = p.replan(None, Y, sched, jax.random.PRNGKey(1))
    u_mean = float(jnp.mean(p.spline.node2u(Y)))
    assert abs(u_mean - TARGET) < 0.1, u_mean


def test_bridge_dial_end_to_end():
    # Full DIAL-MPC == bridge(DialReversePlanner). State is irrelevant to the
    # mock reward, so step_fn is identity; check it runs and warms up to the
    # optimum. NOTE on DIAL semantics: node-0 is the "already-committed" control
    # (pinned during replan), so the FIRST executed action is the cold node-0
    # (~0); the plan refines future nodes and `shift` carries a refined node into
    # node-0, so the executed action warms up to the optimum from step ~1 on.
    p = _planner(Nsample=256)
    ctrl = RecedingHorizonController(
        p,
        step_fn=lambda s, u: s,  # mock real dynamics (reward ignores state)
        n_steps=5,
        n_diffuse_init=25,
        n_diffuse=4,
        collect_plan_vars=True,
    )
    res = ctrl.run(x0=jnp.zeros(4), rng=jax.random.PRNGKey(2))
    assert res.horizon == 5
    assert len(res.actions) == 5
    for a in res.actions:
        assert a.shape == (NU,)
        assert jnp.all(jnp.isfinite(a))
    # first action = cold-start pinned node-0 (~0): DIAL executes-then-refines
    assert abs(float(jnp.mean(res.actions[0]))) < 0.1
    # steady-state executed action (after warm-start + shift) reaches the optimum
    assert abs(float(jnp.mean(res.actions[-1])) - TARGET) < 0.15


def test_bridge_warm_start_keeps_plan_near_optimum():
    # After convergence, the few-step steady replans (warm-started) must KEEP the
    # plan near the optimum rather than drifting back to zeros.
    p = _planner(Nsample=256)
    ctrl = RecedingHorizonController(
        p, step_fn=lambda s, u: s, n_steps=6, n_diffuse_init=25, n_diffuse=4,
        collect_plan_vars=True,
    )
    res = ctrl.run(x0=jnp.zeros(4), rng=jax.random.PRNGKey(3))
    later = [float(jnp.mean(p.spline.node2u(pv))) for pv in res.plan_vars[2:]]
    for m in later:
        assert abs(m - TARGET) < 0.2, later


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
