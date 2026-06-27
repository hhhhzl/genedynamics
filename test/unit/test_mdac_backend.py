"""MDAC backend gates (fedguide CPU, mock rollout — no mjx).

The backend is the mdoc/twogo high-performance pattern (jit + lax.scan
reverse-diffuse, vmap rollout) over DIAL's node-spline substrate, with the MDAC
upgrades composed from EXISTING upstream seams. The load-bearing gate: with every
seam off / NoOp / None, the parallel kernel is byte-identical (<=1e-5) to
DialBackendJax (the degeneration). Plus smoke tests that each seam routes
into the right upstream component."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from genedynamics.solvers.single.dial.backends.dial_jax import DialBackendJax
from genedynamics.solvers.single.mdac.backends.mdac_jax import MdacBackendJax
from genedynamics.core.constraints.action_filters.base import ConstraintFilter


COMMON = dict(
    nu=2, Hnode=4, Hsample=16, Nsample=64, temp_sample=0.06,
    horizon_diffuse_factor=0.9, traj_diffuse_factor=0.5, sigma_scale=1.0,
    action_limit=1.0, ctrl_dt=0.02, Ndiffuse=2, Ndiffuse_init=8, seed=0,
)
TARGET = jnp.array([0.5, -0.3])
X0 = jnp.zeros((1,))


def _mock_rollout(state, us, t0):           # us: (B, Hsample+1, nu) -> (B, Hsample+1)
    return -jnp.sum((us - TARGET) ** 2, axis=-1)


def test_seams_off_equals_dial_byte_identical():
    rng = jax.random.PRNGKey(123)
    dial = DialBackendJax(rollout_fn=_mock_rollout, update_form="weighted_mean", **COMMON)
    mdac = MdacBackendJax(rollout_fn=_mock_rollout, **COMMON)   # transport=None, NoOp filter, no geom
    Yd = dial.plan(X0, rng_key=rng)["actions"]
    Ym = mdac.plan(X0, rng_key=rng)["actions"]
    assert Yd.shape == Ym.shape
    assert np.max(np.abs(Yd - Ym)) < 1e-5, np.max(np.abs(Yd - Ym))


def test_kernel_is_jit_scan_parallel():
    # replan must be a single jit-able lax.scan (the high-perf pattern), and the
    # rollout vmap-ed over Nsample (mock returns batched rewards).
    mdac = MdacBackendJax(rollout_fn=_mock_rollout, **COMMON)
    replan_jit = jax.jit(mdac.replan)
    Y = replan_jit(X0, mdac.init_plan_var(), mdac.make_schedule(mdac.Ndiffuse_init),
                   jax.random.PRNGKey(1))
    assert Y.shape == (COMMON["Hnode"] + 1, COMMON["nu"]) and jnp.all(jnp.isfinite(Y))


def test_transport_seam_routes_to_ddpm():
    from genedynamics.solvers.common.transport import DDPMTransport
    rng = jax.random.PRNGKey(7)
    mdac = MdacBackendJax(rollout_fn=_mock_rollout, transport=DDPMTransport(), **COMMON)
    out = mdac.plan(X0, rng_key=rng)
    assert jnp.all(jnp.isfinite(jnp.asarray(out["actions"])))


def test_geometry_seam_routes_to_sdfmanifold():
    from genedynamics.genemetry.manifold.sdf import SdfManifold
    rng = jax.random.PRNGKey(9)
    mdac = MdacBackendJax(rollout_fn=_mock_rollout, **COMMON)
    # inject a constraint-geometry fn + the genemetry manifold (what a constrained
    # env would supply). a_geom = the node mean itself (a valid (Hnode+1, nu) field).
    mdac.geometry_fn = lambda state, Y, t0: Y
    mdac.manifold = SdfManifold(backend="jax")
    mdac.topk_active, mdac.eps_stab, mdac.geom_gain, mdac.retraction = 2, 1e-6, 1.0, None
    out = mdac.plan(X0, rng_key=rng)
    assert jnp.all(jnp.isfinite(jnp.asarray(out["actions"])))


def test_constraint_filter_seam_is_invoked():
    calls = {"n": 0}

    class _ScaleFilter(ConstraintFilter):
        def apply_actions(self, x0, actions, *, env=None, obstacles=None,
                          schedule_state=None, schedule_params=None, **kw):
            calls["n"] += 1
            return actions * 0.9

        def apply_actions_batch(self, x0, actions, **kw):
            return self.apply_actions(x0, actions, **kw)

    rng = jax.random.PRNGKey(3)
    base = MdacBackendJax(rollout_fn=_mock_rollout, **COMMON).plan(X0, rng_key=rng)["actions"]
    filt = MdacBackendJax(rollout_fn=_mock_rollout, constraint_filter=_ScaleFilter(), **COMMON)
    out = filt.plan(X0, rng_key=rng)["actions"]
    assert calls["n"] > 0                                    # the seam was actually called
    assert np.max(np.abs(out - base)) > 1e-6                 # and changed the result


def test_solver_registered_and_method_flags():
    from genedynamics.core.registry.solvers import get_solver_registry
    from genedynamics.solvers.single.mdac import MDACSolver
    from genedynamics.solvers.single.mdac.core.method_registry import (
        resolve_method, assert_single_flag_ablation,
    )
    assert get_solver_registry().get_class("mdac") is MDACSolver
    assert all(getattr(resolve_method("mdac"), f) for f in resolve_method("mdac").__dataclass_fields__)
    # dial_nostiff = the true all-off anchor (byte-identity); dial is the fair
    # baseline (shares the position-stiffness primitive, MDAC solver seams off).
    assert not any(getattr(resolve_method("dial_nostiff"), f) for f in resolve_method("dial_nostiff").__dataclass_fields__)
    dial = resolve_method("dial")
    assert dial.use_stiffness and not dial.use_tangent_projection and not dial.use_soft_feasibility
    assert_single_flag_ablation("mdac", "mdac_no_tangent")
