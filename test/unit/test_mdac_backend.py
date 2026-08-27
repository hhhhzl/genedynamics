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


def test_steady_schedule_uses_static_shape_with_exact_masked_tail():
    mdac = MdacBackendJax(rollout_fn=_mock_rollout, **COMMON)
    schedule = mdac.make_schedule(mdac.Ndiffuse)
    assert schedule.shape == (
        COMMON["Ndiffuse_init"], COMMON["Hnode"] + 1
    )
    assert bool(jnp.all(jnp.isfinite(schedule[:COMMON["Ndiffuse"]])))
    assert bool(jnp.all(jnp.isnan(schedule[COMMON["Ndiffuse"]:])))

    rng = jax.random.PRNGKey(17)
    warm = mdac.init_plan_var()
    t0 = jnp.float32(0.0)
    masked = mdac._replan_scan(X0, warm, schedule, rng, t0, warm)
    unpadded = mdac._replan_scan(
        X0, warm, schedule[:COMMON["Ndiffuse"]], rng, t0, warm
    )
    np.testing.assert_allclose(masked, unpadded, atol=1e-6, rtol=1e-6)


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


def test_zero_geometry_gate_recovers_raw_weighted_update():
    """Gate=0 must recover the raw DIAL direction even with geometry present."""
    from genedynamics.genemetry.manifold.sdf import SdfManifold

    rng = jax.random.PRNGKey(19)
    raw = MdacBackendJax(rollout_fn=_mock_rollout, **COMMON)
    expected = raw.plan(X0, rng_key=rng)["actions"]

    def gate(value):
        return lambda state, Y, t0: {
            "action": jnp.full((COMMON["nu"],), value),
            "scalar": jnp.float32(value),
            "path": jnp.float32(value),
            "normal": jnp.float32(value),
            "stiffness": jnp.float32(value),
            "force": jnp.float32(value),
        }

    gated = MdacBackendJax(
        rollout_fn=_mock_rollout, geometry_gate_fn=gate(0.0), **COMMON
    )
    gated.geometry_fn = lambda state, Y, t0: Y + 0.1
    gated.manifold = SdfManifold(backend="jax")
    gated.topk_active, gated.eps_stab, gated.geom_gain = 2, 1e-6, 1.0
    actual = gated.plan(X0, rng_key=rng)["actions"]
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    gated_one = MdacBackendJax(
        rollout_fn=_mock_rollout, geometry_gate_fn=gate(1.0), **COMMON
    )
    gated_one.geometry_fn = lambda state, Y, t0: Y + 0.1
    gated_one.manifold = SdfManifold(backend="jax")
    gated_one.topk_active, gated_one.eps_stab, gated_one.geom_gain = 2, 1e-6, 1.0
    projected = gated_one.plan(X0, rng_key=rng)["actions"]
    assert np.max(np.abs(projected - expected)) > 1e-6


def test_controllability_gate_does_not_disable_clean_projection():
    """Controllability reliability gates B-correction, not clean geometry."""
    from genedynamics.genemetry.manifold.sdf import SdfManifold

    def zero_gate(state, Y, t0):
        return {
            "action": jnp.zeros((COMMON["nu"],)),
            "scalar": jnp.float32(0.0),
            "path": jnp.float32(0.0),
            "normal": jnp.float32(0.0),
            "stiffness": jnp.float32(0.0),
            "force": jnp.float32(0.0),
        }

    rng = jax.random.PRNGKey(29)
    raw = MdacBackendJax(rollout_fn=_mock_rollout, **COMMON)
    expected_raw = raw.plan(X0, rng_key=rng)["actions"]
    gated = MdacBackendJax(
        rollout_fn=_mock_rollout, geometry_gate_fn=zero_gate, **COMMON
    )
    gated.geometry_fn = lambda state, Y, t0: Y + 0.1
    gated.manifold = SdfManifold(backend="jax")
    gated.topk_active, gated.eps_stab, gated.geom_gain = 2, 1e-6, 1.0
    gated._gate_controllability_only = True
    projected = gated.plan(X0, rng_key=rng)["actions"]
    assert np.max(np.abs(projected - expected_raw)) > 1e-6


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
        resolve_method, assert_single_flag_ablation, diff_flags,
    )
    assert get_solver_registry().get_class("mdac") is MDACSolver
    full = resolve_method("mdac")
    assert full.use_tangent_projection and full.use_retraction
    assert not full.use_horizon_geometry and not full.use_geometry_gate
    # dial_nostiff = the true all-off anchor (byte-identity); dial is the fair
    # baseline (shares the position-stiffness primitive, MDAC solver seams off).
    assert not any(getattr(resolve_method("dial_nostiff"), f) for f in resolve_method("dial_nostiff").__dataclass_fields__)
    dial = resolve_method("dial")
    assert dial.use_stiffness and not dial.use_tangent_projection and not dial.use_soft_feasibility
    assert_single_flag_ablation("mdac", "mdac_no_tangent")
    assert_single_flag_ablation("mdac", "mdac_no_retraction")
    assert_single_flag_ablation("mdac", "mdac_position_only")
    assert not resolve_method("mdac_position_only").use_force_manifold
    horizon = resolve_method("mdac_horizon")
    scalar = resolve_method("mdac_scalar_gate")
    component = resolve_method("mdac_component_gate")
    realization = resolve_method("mdac_realization")
    realization_gate = resolve_method("mdac_realization_gate")
    controllable = resolve_method("mdac_controllable")
    assert horizon.use_horizon_geometry and not horizon.use_geometry_gate
    assert scalar.use_horizon_geometry and scalar.use_geometry_gate
    assert not scalar.component_geometry_gate
    assert component.use_horizon_geometry and component.use_geometry_gate
    assert component.component_geometry_gate
    assert realization.use_horizon_geometry
    assert realization.use_realization_compensation
    assert not realization.use_geometry_gate
    assert realization_gate.use_realization_compensation
    assert realization_gate.component_geometry_gate
    assert controllable.use_horizon_geometry
    assert controllable.use_controllability_geometry
    assert not controllable.use_geometry_gate
    current = resolve_method("mdac_controllable_gate")
    surface_ablations = {
        "mdac_horizon": ("use_controllability_geometry",),
        "mdac_controllable_no_retraction": ("use_retraction",),
        "mdac_controllable_no_stiffness": ("use_stiffness",),
        "mdac_controllable_euclid_stiffness": ("log_spd_stiffness",),
        "mdac_controllable_gate_no_tangent": ("use_tangent_projection",),
    }
    for ablation, expected_difference in surface_ablations.items():
        base = controllable if ablation == "mdac_horizon" else current
        assert diff_flags(base, resolve_method(ablation)) == expected_difference
