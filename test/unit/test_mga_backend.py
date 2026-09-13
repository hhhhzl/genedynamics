"""MGA backend gates (fedguide CPU, mock rollout — no mjx).

The backend is the mdoc/twogo high-performance pattern (jit + lax.scan
reverse-diffuse, vmap rollout) over DIAL's node-spline substrate, with the MGA
upgrades composed from EXISTING upstream seams. The load-bearing gate: with every
seam off / NoOp / None, the parallel kernel is byte-identical (<=1e-5) to
DialBackendJax (the degeneration). Plus smoke tests that each seam routes
into the right upstream component."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from genedynamics.solvers.single.dial.backends.dial_jax import DialBackendJax
from genedynamics.solvers.single.mga.backends.mga_jax import MgaBackendJax
from genedynamics.core.constraints.action_filters.base import ConstraintFilter
from genedynamics.core.types import ExecutionRejected


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
    mga = MgaBackendJax(rollout_fn=_mock_rollout, **COMMON)   # transport=None, NoOp filter, no geom
    Yd = dial.plan(X0, rng_key=rng)["actions"]
    Ym = mga.plan(X0, rng_key=rng)["actions"]
    assert Yd.shape == Ym.shape
    assert np.max(np.abs(Yd - Ym)) < 1e-5, np.max(np.abs(Yd - Ym))


def test_explicit_zero_shift_preserves_no_prior_incumbent_and_emergency():
    backend = MgaBackendJax(rollout_fn=_mock_rollout, **COMMON)
    backend.receding_shift_mode = "zero"
    backend.prior_fallback_mode = "receding_incumbent"
    backend.emergency_active_fn = lambda nodes: jnp.all(nodes[:, -1] < -0.99)
    ordinary = jnp.linspace(-0.2, 0.7, 10).reshape(5, 2)
    backend.prior = object()
    backend.use_rl_prior = True
    with_prior = jax.jit(backend.shift)(ordinary)
    backend.prior = None
    without_prior = backend.shift(ordinary)
    np.testing.assert_array_equal(with_prior, without_prior)
    np.testing.assert_array_equal(with_prior, backend.spline.shift_nodes(ordinary))

    emergency = ordinary.at[:, -1].set(-1.0)
    np.testing.assert_array_equal(
        backend.shift(emergency), backend.spline.shift_nodes_terminal_hold(emergency)
    )


@pytest.mark.parametrize("prior_active", [False, True])
@pytest.mark.parametrize("fallback_mode", ["rl", "receding_incumbent"])
@pytest.mark.parametrize("emergency", [False, True])
def test_legacy_shift_keeps_frozen_arm_prior_behavior(
    prior_active, fallback_mode, emergency,
):
    """The H1 opt-in must not change any branch of the frozen default shift."""
    backend = MgaBackendJax(rollout_fn=_mock_rollout, **COMMON)
    assert backend.receding_shift_mode == "legacy"
    backend.prior_fallback_mode = fallback_mode
    backend.prior = object() if prior_active else None
    backend.use_rl_prior = True
    backend.emergency_active_fn = lambda nodes: jnp.all(nodes[:, -1] < -0.99)
    ordinary = jnp.linspace(-0.2, 0.7, 10).reshape(5, 2)
    initial = ordinary.at[:, -1].set(-1.0) if emergency else ordinary

    def frozen_shift(nodes):
        # Reference is the complete pre-option implementation, not the new
        # explicit zero/terminal_hold modes we are trying to keep isolated.
        if prior_active and fallback_mode == "receding_incumbent":
            return backend.spline.shift_nodes_terminal_hold(nodes)
        if fallback_mode == "receding_incumbent":
            return jnp.where(
                backend.emergency_active_fn(nodes),
                backend.spline.shift_nodes_terminal_hold(nodes),
                backend.spline.shift_nodes(nodes),
            )
        return backend.spline.shift_nodes(nodes)

    expected, actual = initial, initial
    reference_jit, current_jit = jax.jit(frozen_shift), jax.jit(backend.shift)
    for _ in range(4):
        expected, actual = reference_jit(expected), current_jit(actual)
        np.testing.assert_array_equal(actual, expected)


def test_kernel_is_jit_scan_parallel():
    # replan must be a single jit-able lax.scan (the high-perf pattern), and the
    # rollout vmap-ed over Nsample (mock returns batched rewards).
    mga = MgaBackendJax(rollout_fn=_mock_rollout, **COMMON)
    replan_jit = jax.jit(mga.replan)
    Y = replan_jit(X0, mga.init_plan_var(), mga.make_schedule(mga.Ndiffuse_init),
                   jax.random.PRNGKey(1))
    assert Y.shape == (COMMON["Hnode"] + 1, COMMON["nu"]) and jnp.all(jnp.isfinite(Y))


def test_stepwise_diffusion_matches_fused_reverse_updates():
    """The CPU compilation boundary must not change update or RNG order."""
    fused = MgaBackendJax(rollout_fn=_mock_rollout, **COMMON)
    stepwise = MgaBackendJax(
        rollout_fn=_mock_rollout, stepwise_diffusion=True, **COMMON
    )
    warm = fused.init_plan_var()
    schedule = fused.make_schedule(2)
    key = jax.random.PRNGKey(29)
    expected, _ = fused.replan_with_info(
        X0, warm, schedule, key, t0=jnp.float32(3.0)
    )
    actual, _ = stepwise.replan_with_info(
        X0, warm, schedule, key, t0=jnp.float32(3.0)
    )
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_warm_start_prior_is_external_to_refinement_graph():
    """The CPU prior changes the centre, not the model-based rollout graph."""
    from types import SimpleNamespace

    backend = MgaBackendJax(rollout_fn=_mock_rollout, **COMMON)
    proposal = jnp.full_like(backend.init_plan_var(), 0.4)
    backend.prior = SimpleNamespace(warm_start=lambda state: proposal)
    backend.use_rl_prior = True
    backend.prior_mode = "warm_start"
    backend.prior_lambda_shift = 0.25
    backend.prior_acceptance = False
    captured = {}

    def capture(state, warm_start, schedule, rng, t0, prior):
        captured["warm_start"] = warm_start
        captured["prior_argument"] = prior
        return warm_start

    backend._replan_scan_jit = capture
    selected, _ = backend.replan_with_info(
        X0, backend.init_plan_var(), backend.make_schedule(2),
        jax.random.PRNGKey(5),
    )
    expected = 0.75 * proposal
    np.testing.assert_array_equal(captured["warm_start"], expected)
    np.testing.assert_array_equal(captured["prior_argument"], proposal)
    np.testing.assert_array_equal(selected, expected)
    assert not backend._prior_guided


def test_sequential_candidate_certificates_match_vmap():
    candidates = jnp.arange(24, dtype=jnp.float32).reshape(3, 4, 2)
    score = lambda x: (jnp.mean(x), jnp.max(jnp.abs(x), axis=0))
    parallel = MgaBackendJax(rollout_fn=_mock_rollout, **COMMON)
    sequential = MgaBackendJax(
        rollout_fn=_mock_rollout, candidate_rollout_batch_size=1, **COMMON
    )
    expected = parallel._map_candidate_scores(score, candidates)
    actual = sequential._map_candidate_scores(score, candidates)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_stepwise_acceptance_materialization_matches_fused_acceptance():
    """Splitting the CPU certificate boundary must not change selection."""
    fused = _context_backend()
    stepwise = _context_backend(
        candidate_rollout_batch_size=1, stepwise_acceptance=True
    )
    state = _ContextToyTask.reset()
    fallback = jnp.zeros((5, 2)).at[:, 0].set(0.3)
    refined = fallback.at[:, 0].set(-0.5)
    emergency = fused.emergency_plan_fn(state, fallback)
    expected, expected_info = fused._accept_refinement_jit(
        state, fallback, refined, jnp.float32(1), None, emergency,
        fallback_mode=jnp.int32(0),
    )
    actual, actual_info = stepwise._accept_refinement_jit(
        state, fallback, refined, jnp.float32(1), None, emergency,
        fallback_mode=jnp.int32(0),
    )
    np.testing.assert_array_equal(actual, expected)
    for key in expected_info:
        if key == "emergency_validation_applicable":
            continue
        np.testing.assert_array_equal(actual_info[key], expected_info[key])


def test_lazy_emergency_scoring_skips_unselectable_unload_certificate():
    backend = _context_backend(
        candidate_rollout_batch_size=1,
        stepwise_acceptance=True,
        lazy_emergency_scoring=True,
    )
    backend._emergency_candidate_scores_jit = lambda *args: (_ for _ in ()).throw(
        AssertionError("safe NORMAL candidates must not compile UNLOAD scoring")
    )
    state = _ContextToyTask.reset()
    fallback = jnp.zeros((5, 2)).at[:, 0].set(0.3)
    refined = fallback.at[:, 0].set(0.2)
    emergency = backend.emergency_plan_fn(state, fallback)
    selected, info = backend._accept_refinement_jit(
        state, fallback, refined, jnp.float32(1), None, emergency,
        fallback_mode=jnp.int32(0),
    )
    assert selected.shape == fallback.shape
    assert float(info["emergency_validation_candidate_count"]) == 0.0
    assert float(info["emergency_validation_applicable"]) == 0.0


def test_lazy_emergency_scoring_prefers_task_host_certificate():
    class HostCertificateTask(_ContextToyTask):
        def __init__(self):
            super().__init__()
            self.host_calls = 0

        @property
        def mga_execution_context(self):
            context = super().mga_execution_context

            def score_emergency(state, actions, aug_lambda, aug_rho):
                del aug_lambda, aug_rho
                self.host_calls += 1
                return self.emergency_sequence_score_risk(state, actions)

            return {**context, "score_emergency_host": score_emergency}

    task = HostCertificateTask()
    backend = _context_backend(
        model=task, execution=task,
        candidate_rollout_batch_size=1,
        stepwise_acceptance=True,
        lazy_emergency_scoring=True,
    )
    backend._emergency_candidate_scores_jit = lambda *args: (_ for _ in ()).throw(
        AssertionError("task host certificate must replace the duplicate JIT")
    )
    state = task.reset()
    unsafe = jnp.zeros((5, 2)).at[:, 0].set(-0.5)
    emergency = backend.emergency_plan_fn(state, unsafe)
    _, info = backend._accept_refinement_jit(
        state, unsafe, unsafe, jnp.float32(1), None, emergency,
        fallback_mode=jnp.int32(0),
    )
    assert int(info["execution_mode"]) == 1
    assert task.host_calls == 2


def test_steady_schedule_uses_static_shape_with_exact_masked_tail():
    mga = MgaBackendJax(rollout_fn=_mock_rollout, **COMMON)
    schedule = mga.make_schedule(mga.Ndiffuse)
    assert schedule.shape == (
        COMMON["Ndiffuse_init"], COMMON["Hnode"] + 1
    )
    assert bool(jnp.all(jnp.isfinite(schedule[:COMMON["Ndiffuse"]])))
    assert bool(jnp.all(jnp.isnan(schedule[COMMON["Ndiffuse"]:])))

    rng = jax.random.PRNGKey(17)
    warm = mga.init_plan_var()
    t0 = jnp.float32(0.0)
    masked = mga._replan_scan(X0, warm, schedule, rng, t0, warm)
    unpadded = mga._replan_scan(
        X0, warm, schedule[:COMMON["Ndiffuse"]], rng, t0, warm
    )
    np.testing.assert_allclose(masked, unpadded, atol=1e-6, rtol=1e-6)


def test_transport_seam_routes_to_ddpm():
    from genedynamics.solvers.common.transport import DDPMTransport
    rng = jax.random.PRNGKey(7)
    mga = MgaBackendJax(rollout_fn=_mock_rollout, transport=DDPMTransport(), **COMMON)
    out = mga.plan(X0, rng_key=rng)
    assert jnp.all(jnp.isfinite(jnp.asarray(out["actions"])))


def test_geometry_seam_routes_to_sdfmanifold():
    from genedynamics.genemetry.manifold.sdf import SdfManifold
    rng = jax.random.PRNGKey(9)
    mga = MgaBackendJax(rollout_fn=_mock_rollout, **COMMON)
    # inject a constraint-geometry fn + the genemetry manifold (what a constrained
    # env would supply). a_geom = the node mean itself (a valid (Hnode+1, nu) field).
    mga.geometry_fn = lambda state, Y, t0: Y
    mga.manifold = SdfManifold(backend="jax")
    mga.topk_active, mga.eps_stab, mga.geom_gain, mga.retraction = 2, 1e-6, 1.0, None
    out = mga.plan(X0, rng_key=rng)
    assert jnp.all(jnp.isfinite(jnp.asarray(out["actions"])))


def test_zero_geometry_gate_recovers_raw_weighted_update():
    """Gate=0 must recover the raw DIAL direction even with geometry present."""
    from genedynamics.genemetry.manifold.sdf import SdfManifold

    rng = jax.random.PRNGKey(19)
    raw = MgaBackendJax(rollout_fn=_mock_rollout, **COMMON)
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

    gated = MgaBackendJax(
        rollout_fn=_mock_rollout, geometry_gate_fn=gate(0.0), **COMMON
    )
    gated.geometry_fn = lambda state, Y, t0: Y + 0.1
    gated.manifold = SdfManifold(backend="jax")
    gated.topk_active, gated.eps_stab, gated.geom_gain = 2, 1e-6, 1.0
    actual = gated.plan(X0, rng_key=rng)["actions"]
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    gated_one = MgaBackendJax(
        rollout_fn=_mock_rollout, geometry_gate_fn=gate(1.0), **COMMON
    )
    gated_one.geometry_fn = lambda state, Y, t0: Y + 0.1
    gated_one.manifold = SdfManifold(backend="jax")
    gated_one.topk_active, gated_one.eps_stab, gated_one.geom_gain = 2, 1e-6, 1.0
    projected = gated_one.plan(X0, rng_key=rng)["actions"]
    assert np.max(np.abs(projected - expected)) > 1e-6


def test_optional_update_gate_freezes_unobservable_blocks_at_incumbent():
    """A task may preserve a trusted center without changing legacy gate=0."""
    from genedynamics.genemetry.manifold.sdf import SdfManifold

    def gate(state, Y, t0):
        del state, Y, t0
        return {
            "action": jnp.zeros((COMMON["nu"],)),
            "update_action": jnp.zeros((COMMON["nu"],)),
            "scalar": jnp.float32(0.0),
            "path": jnp.float32(0.0),
            "normal": jnp.float32(0.0),
            "stiffness": jnp.float32(0.0),
            "force": jnp.float32(0.0),
        }

    backend = MgaBackendJax(
        rollout_fn=_mock_rollout, geometry_gate_fn=gate, **COMMON
    )
    backend.geometry_fn = lambda state, Y, t0: Y + 0.1
    backend.manifold = SdfManifold(backend="jax")
    backend.topk_active, backend.eps_stab, backend.geom_gain = 2, 1e-6, 1.0
    initial = backend.init_plan_var()
    actual = backend.replan(
        X0, initial, backend.make_schedule(COMMON["Ndiffuse"]),
        jax.random.PRNGKey(19),
    )
    np.testing.assert_array_equal(actual, initial)


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
    raw = MgaBackendJax(rollout_fn=_mock_rollout, **COMMON)
    expected_raw = raw.plan(X0, rng_key=rng)["actions"]
    gated = MgaBackendJax(
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
    base = MgaBackendJax(rollout_fn=_mock_rollout, **COMMON).plan(X0, rng_key=rng)["actions"]
    filt = MgaBackendJax(rollout_fn=_mock_rollout, constraint_filter=_ScaleFilter(), **COMMON)
    out = filt.plan(X0, rng_key=rng)["actions"]
    assert calls["n"] > 0                                    # the seam was actually called
    assert np.max(np.abs(out - base)) > 1e-6                 # and changed the result


def test_solver_registered_and_method_flags():
    from genedynamics.core.registry.solvers import get_solver_registry
    from genedynamics.solvers.single.mga import MGASolver
    from genedynamics.solvers.single.mga.core.method_registry import (
        resolve_method, assert_single_flag_ablation, diff_flags,
    )
    assert get_solver_registry().get_class("mga") is MGASolver
    base = resolve_method("mga_base")
    assert base.use_tangent_projection and base.use_retraction
    assert not base.use_horizon_geometry and not base.use_geometry_gate
    # The public paper name resolves to the current frozen controller, while a
    # historical name is accepted only at the centralized compatibility seam.
    assert resolve_method("mga") == resolve_method("mga_controllable_gate")
    assert resolve_method("mdac") == base
    # dial_nostiff = the true all-off anchor (byte-identity); dial is the fair
    # baseline (shares the position-stiffness primitive, MGA solver seams off).
    assert not any(getattr(resolve_method("dial_nostiff"), f) for f in resolve_method("dial_nostiff").__dataclass_fields__)
    dial = resolve_method("dial")
    assert dial.use_stiffness and not dial.use_tangent_projection and not dial.use_soft_feasibility
    assert_single_flag_ablation("mga_base", "mga_no_tangent")
    assert_single_flag_ablation("mga_base", "mga_no_retraction")
    assert_single_flag_ablation("mga_base", "mga_position_only")
    assert not resolve_method("mga_position_only").use_force_manifold
    horizon = resolve_method("mga_horizon")
    scalar = resolve_method("mga_scalar_gate")
    component = resolve_method("mga_component_gate")
    realization = resolve_method("mga_realization")
    realization_gate = resolve_method("mga_realization_gate")
    controllable = resolve_method("mga_controllable")
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
    current = resolve_method("mga_controllable_gate")
    surface_ablations = {
        "mga_horizon": ("use_controllability_geometry",),
        "mga_controllable_no_retraction": ("use_retraction",),
        "mga_controllable_no_stiffness": ("use_stiffness",),
        "mga_controllable_euclid_stiffness": ("log_spd_stiffness",),
        "mga_controllable_gate_no_tangent": ("use_tangent_projection",),
    }
    for ablation, expected_difference in surface_ablations.items():
        base = controllable if ablation == "mga_horizon" else current
        assert diff_flags(base, resolve_method(ablation)) == expected_difference


class _ContextToyTask:
    """Tiny mode-sensitive transition; no robot assets or physics."""
    def __init__(self, anchor=2.0, stiffness_scale=1.0):
        self.anchor = anchor
        self.stiffness_scale = stiffness_scale

    @property
    def mga_execution_context(self):
        return {
            "schema_version": 1, "normal_mode": 0, "emergency_mode": 1,
            "prepare_state": self.prepare_state,
            "mode_from_state": lambda state: state["mode"],
            "step": self.step,
        }

    @staticmethod
    def reset(mode=0):
        return {
            "mode": jnp.int32(mode), "request": jnp.int32(-1),
            "anchor": jnp.float32(0), "normal": jnp.float32(1),
            "entry_prepared": jnp.asarray(mode == 1),
            "entry_geometry_valid": jnp.asarray(mode == 1),
            # Fixture for actual prev_action[s_slice], independent of toy's
            # two-dimensional planning coordinates.
            "prev_stiffness_raw": jnp.zeros(6, jnp.float32),
            "stiffness_raw": jnp.zeros(6, jnp.float32),
            "stiffness_matrix": jnp.eye(3, dtype=jnp.float32),
            "applied_stiffness_matrix": jnp.eye(3, dtype=jnp.float32),
            "stiffness_valid": jnp.asarray(mode == 1),
            "steps": jnp.int32(0), "initial_violation": jnp.float32(0),
        }

    def _decode_stiffness(self, raw):
        return jnp.diag(jnp.exp(self.stiffness_scale * raw[:3]))

    def _capture_entry(self, state):
        raw = state["prev_stiffness_raw"]
        matrix = self._decode_stiffness(raw)
        return {
            "anchor": jnp.float32(self.anchor), "normal": jnp.float32(1),
            "entry_geometry_valid": jnp.asarray(True),
            "stiffness_raw": raw, "stiffness_matrix": matrix,
            "stiffness_valid": (
                jnp.all(jnp.isfinite(raw) & (jnp.abs(raw) <= 1))
                & jnp.all(jnp.isfinite(matrix))
            ),
        }

    def prepare_state(self, state, mode):
        # entry_prepared is independent of geometric validity. In particular,
        # invalid-but-sealed entries cannot become "valid" in nominal geometry.
        active = (state["mode"] == 1) | (state["request"] == 1)
        reuse = active & state["entry_prepared"]
        capture = (
            (mode == 1) & (state["mode"] == 0) & (state["request"] != 1)
            & (~state["entry_prepared"])
        )
        entry = jax.lax.cond(
            capture, self._capture_entry,
            lambda current: {
                name: current[name]
                for name in ("anchor", "normal", "entry_geometry_valid",
                             "stiffness_raw", "stiffness_matrix", "stiffness_valid")
            },
            state,
        )
        prepared = capture | reuse
        geometry_valid = jnp.where(
            capture, entry["entry_geometry_valid"],
            reuse & state["entry_geometry_valid"],
        )
        result = {**state, "request": jnp.int32(mode)}
        for name in ("anchor", "normal", "stiffness_raw", "stiffness_matrix"):
            result[name] = jnp.where(
                (mode == 1) & capture, entry[name], state[name]
            )
        for name, value in (
            ("entry_prepared", prepared),
            ("entry_geometry_valid", geometry_valid),
            ("stiffness_valid", jnp.where(
                capture, entry["stiffness_valid"], reuse & state["stiffness_valid"]
            )),
        ):
            # NORMAL preserves committed history, but cancels an entry that
            # was only prepared speculatively and never physically executed.
            result[name] = jnp.where(
                mode == 1, value,
                jnp.where(state["mode"] == 1, state[name], jnp.asarray(False)),
            )
        return result

    def step(self, state, action, mode=0):
        del action
        state = self.prepare_state(state, jnp.int32(mode))
        return {
            **state, "mode": jnp.int32(mode), "request": jnp.int32(-1),
            "entry_prepared": (mode == 1) & state["entry_prepared"],
            "entry_geometry_valid": (mode == 1) & state["entry_geometry_valid"],
            "stiffness_valid": (mode == 1) & state["stiffness_valid"],
            "applied_stiffness_matrix": jnp.where(
                mode == 1, state["stiffness_matrix"],
                self._decode_stiffness(state["prev_stiffness_raw"]),
            ),
            "steps": state["steps"] + 1,
        }

    def sequence_score_risk(self, state, us, *unused):
        safe = (jnp.mean(us[:, 0]) >= 0.2) & (jnp.mean(us[:, 0]) <= 0.8)
        risk = jnp.zeros(4).at[0].set(
            (~safe) | (state["request"] != 0) | (state["initial_violation"] > 0)
        )
        return -jnp.mean((us[:, 0] - 0.3) ** 2), risk

    def emergency_sequence_score_risk(self, state, us, *unused):
        next_state = self.step(state, us[0], 1)
        risk = jnp.zeros(4).at[0].set(
            (state["request"] != 1) | (state["initial_violation"] > 0)
            | jnp.any(next_state["anchor"] != state["anchor"])
        )
        known = (state["entry_prepared"] & state["entry_geometry_valid"]
                 & state["stiffness_valid"])
        # An absent/invalid context is NOT a measured force violation label.
        risk = jnp.where(known, risk, jnp.full((4,), jnp.inf))
        return 10.0 + jnp.sum(next_state["anchor"]), risk

    @staticmethod
    def sequence_risk_is_safe(risk):
        return jnp.all(risk[:3] <= 0)

    @staticmethod
    def emergency_plan(state, nodes, t0=0):
        return jnp.zeros_like(nodes).at[:, -1].set(-1)

    @staticmethod
    def emergency_plan_is_active(nodes):
        raise AssertionError("explicit modes must not inspect numeric action patterns")


def _context_backend(
    *, acceptance=True, prior_mode="guided", model=None, execution=None,
    **backend_kwargs,
):
    from types import SimpleNamespace
    model = _ContextToyTask() if model is None else model
    execution = model if execution is None else execution
    solver = SimpleNamespace(
        config={}, nu=2, seed=0, dynamics=model, execution_env=execution,
        prior_acceptance=acceptance, prior_fallback_mode="receding_incumbent",
        prior_mode=prior_mode,
        _rollout_fn=_mock_rollout, _step_fn=lambda state, action: state,
    )
    return MgaBackendJax(solver=solver, **COMMON, **backend_kwargs)


def test_context_fallback_scores_its_mode_and_preserves_measured_entry_anchor():
    backend = _context_backend(model=_ContextToyTask(2), execution=_ContextToyTask(7))
    state, nodes = _ContextToyTask.reset(), jnp.zeros((5, 2))
    emergency = backend.emergency_plan_fn(state, nodes)
    selected, info = backend._accept_refinement(
        state, emergency, nodes, jnp.float32(1), emergency=emergency,
        fallback_mode=jnp.int32(1),
    )
    assert int(info["execution_mode"]) == 1
    assert float(info["prior_score_fallback"]) == 17.0  # execution anchor, not model's 2
    assert float(info["incumbent_revalidated_safe"]) == 1
    assert float(info["prior_comparison_applicable"]) == 0
    committed = backend.execution_step(state, selected[0], info)
    assert int(committed["mode"]) == 1 and int(committed["request"]) == -1
    assert float(committed["anchor"]) == 7.0
    assert int(state["mode"]) == 0 and float(state["anchor"]) == 0.0


def test_context_cold_task_initializer_is_revalidated_before_emergency():
    """A certified cold initializer is safer than an unavailable UNLOAD entry."""
    backend = _context_backend()
    state = _ContextToyTask.reset()
    fallback = jnp.zeros((5, 2)).at[:, 0].set(0.3)
    refined = fallback.at[:, 0].set(-0.5)
    emergency = backend.emergency_plan_fn(state, fallback)
    selected, info = backend._accept_refinement(
        state, fallback, refined, jnp.float32(0), emergency=emergency,
        fallback_mode=jnp.int32(0),
    )
    np.testing.assert_array_equal(selected, fallback)
    assert float(info["incumbent_revalidated_safe"]) == 1.0
    assert float(info["emergency_selected"]) == 0.0
    assert int(info["execution_mode"]) == 0


def test_context_selects_highest_scoring_safe_task_emergency_from_bank():
    """A task may expose finite emergency alternatives without weakening safety."""
    backend = _context_backend()
    state = _ContextToyTask.reset()
    unsafe = jnp.zeros((5, 2)).at[:, 0].set(-0.5)
    emergency_bank = jnp.stack([
        jnp.zeros((5, 2)).at[:, 0].set(-0.2),
        jnp.zeros((5, 2)).at[:, 0].set(0.3),
        jnp.zeros((5, 2)).at[:, 0].set(0.6),
    ])

    def emergency_score(_state, actions, *_unused):
        value = jnp.mean(actions[:, 0])
        risk = jnp.zeros(4).at[1].set(value < 0.2)
        return -jnp.square(value - 0.3), risk

    backend.emergency_score_risk_fn = emergency_score
    selected, info = backend._accept_refinement(
        state, unsafe, unsafe, jnp.float32(1), emergency=emergency_bank,
        fallback_mode=jnp.int32(0),
    )
    np.testing.assert_allclose(selected, emergency_bank[1], atol=1e-6)
    assert float(info["emergency_selected"]) == 1.0
    assert float(info["emergency_revalidated_safe"]) == 1.0
    assert float(info["emergency_unrecoverable"]) == 0.0
    assert float(info["emergency_validation_candidate_count"]) == 4.0
    np.testing.assert_array_equal(info["prior_risk_emergency"], jnp.zeros(4))


def test_context_normal_refinement_does_not_inherit_committed_unload():
    backend = _context_backend()
    state = _ContextToyTask.reset(mode=1)
    state = {**state, "request": jnp.int32(1), "anchor": jnp.float32(2)}
    emergency = backend.emergency_plan_fn(state, jnp.zeros((5, 2)))
    # nu=-1 is an ordinary candidate here, despite matching the legacy pattern.
    normal = emergency.at[:, 0].set(0.3)
    selected, info = backend._accept_refinement(
        state, emergency, normal, jnp.float32(1), emergency=emergency,
        fallback_mode=jnp.int32(1),
    )
    np.testing.assert_array_equal(selected, normal)
    assert int(info["execution_mode"]) == 0
    assert float(info["emergency_fallback_recovery"]) == 1
    assert float(info["prior_predicted_improvement"]) == 0  # N/A storage only
    assert float(info["prior_comparison_applicable"]) == 0
    committed = backend.execution_step(state, selected[0], info)
    assert int(committed["mode"]) == 0 and int(committed["request"]) == -1


@pytest.mark.parametrize("bad_mode", [jnp.array([1]), jnp.float32(1), jnp.int32(2)])
def test_context_execution_rejects_bad_mode_and_lost_info(bad_mode):
    backend = _context_backend()
    state = _ContextToyTask.reset()
    with pytest.raises(ValueError):
        backend.execution_step(state, jnp.zeros(2), {"execution_mode": bad_mode})
    with pytest.raises(ValueError, match="lost"):
        backend.execution_step(state, jnp.zeros(2), {})


def test_context_shift_uses_explicit_dynamic_mode_and_never_float_identity():
    backend = _context_backend()
    backend.receding_shift_mode = "zero"
    nodes = jnp.linspace(-0.2, 0.7, 10).reshape(5, 2)
    shift = jax.jit(backend._shift_with_mode)
    for mode in (0, 1, 0):
        expected = (
            backend.spline.shift_nodes_terminal_hold(nodes)
            if mode else backend.spline.shift_nodes(nodes)
        )
        np.testing.assert_array_equal(shift(nodes, jnp.int32(mode)), expected)
        backend.after_step(None, None, _ContextToyTask.reset(mode))
        np.testing.assert_array_equal(backend.shift(nodes), expected)
    with pytest.raises(ValueError, match="host shift"):
        jax.jit(backend.shift)(nodes)
    backend.after_step(None, None, _ContextToyTask.reset(1))
    backend.init_plan_var()  # A new episode cannot retain the previous shift mode.
    np.testing.assert_array_equal(backend.shift(nodes), backend.spline.shift_nodes(nodes))


def test_context_certified_shift_keeps_the_previously_scored_backup_action():
    backend = _context_backend()
    backend.receding_shift_mode = "certified_terminal_hold"
    nodes = jnp.linspace(-0.7, 0.8, 10).reshape(5, 2)
    dense = backend.spline.node2u(nodes)
    for mode in (0, 1):
        shifted = backend._shift_with_mode(nodes, jnp.int32(mode))
        np.testing.assert_array_equal(
            backend.first_action(shifted), dense[1]
        )


def test_context_atacom_replacement_restores_normal_identity():
    backend = _context_backend()
    backend.prior_atacom_default = True
    state, nodes = _ContextToyTask.reset(1), jnp.zeros((5, 2))
    emergency = backend.emergency_plan_fn(state, nodes)
    atacom = nodes.at[:, 0].set(0.3)
    selected, info = backend._accept_refinement(
        state, emergency, nodes, jnp.float32(1), atacom, emergency,
        fallback_mode=jnp.int32(1),
    )
    np.testing.assert_array_equal(selected, atacom)
    assert int(info["execution_mode"]) == 0
    assert float(info["atacom_incumbent_selected"]) == 1


@pytest.mark.parametrize("expert_case", ["safe", "unsafe", "unsupported"])
def test_context_additive_recovery_is_safety_first_not_cross_horizon_score(expert_case):
    from types import SimpleNamespace
    backend = _context_backend(prior_mode="additive")
    state, initial = _ContextToyTask.reset(), jnp.zeros((5, 2))
    expert = initial.at[:, 0].set(-0.5 if expert_case == "unsafe" else 0.3)
    backend.prior = SimpleNamespace(warm_start=lambda _: expert)
    backend._replan_scan_jit = lambda *args: initial  # unsafe Gaussian choice
    if expert_case == "unsupported":
        backend.reliability_sequence_feature_fn = lambda state, us: jnp.mean(us, axis=0)
        backend.reliability_model = SimpleNamespace(
            predict_upper=lambda features: jnp.zeros((*features.shape[:-1], 4)),
            support_score=lambda features: jnp.full(features.shape[:-1], 2.0),
        )
    selected, info = backend.replan_with_info(
        state, initial, backend.make_schedule(1), jax.random.PRNGKey(3), t0=1,
    )
    if expert_case == "safe":
        np.testing.assert_array_equal(selected, expert)
        assert int(info["execution_mode"]) == 0
        assert float(info["additive_prior_selected"]) == 1
        assert float(info["additive_prior_comparison_applicable"]) == 0
        assert float(info["emergency_fallback_recovery"]) == 1
        assert float(info["additive_prior_score"]) < float(info["additive_gaussian_score"])
    else:
        assert int(info["execution_mode"]) == 1
        assert float(info["additive_prior_selected"]) == 0
    import json
    json.dumps({key: np.asarray(value).tolist() for key, value in info.items()},
               allow_nan=False)


def test_context_disabled_acceptance_preserves_no_prior_degenerate_normal_replan():
    backend = _context_backend(acceptance=False)
    assert backend.execution_step is None and backend._execution_context is None
    reference = MgaBackendJax(rollout_fn=_mock_rollout, **COMMON)
    initial, key = backend.init_plan_var(), jax.random.PRNGKey(11)
    actual = backend.replan(
        _ContextToyTask.reset(), initial, backend.make_schedule(2), key
    )
    expected = reference.replan(X0, initial, reference.make_schedule(2), key)
    np.testing.assert_array_equal(actual, expected)
    # These public APIs remain available when no route can choose UNLOAD.
    assert backend.plan(X0, key)["actions"].shape == (17, 2)


@pytest.mark.parametrize("which", ["missing", "schema", "callable"])
def test_context_rejects_partial_model_execution_protocol(which):
    class Broken(_ContextToyTask):
        @property
        def mga_execution_context(self):
            value = super().mga_execution_context
            if which == "missing":
                return None
            if which == "schema":
                value["schema_version"] = 2
            else:
                value["step"] = None
            return value
    with pytest.raises((TypeError, ValueError), match="execution context"):
        _context_backend(execution=Broken())


def test_context_action_only_planning_api_fails_closed():
    backend = _context_backend()
    with pytest.raises(ValueError, match="run_receding"):
        backend.plan(_ContextToyTask.reset())
    with pytest.raises(ValueError, match="replan_with_info"):
        backend.replan(
            _ContextToyTask.reset(), backend.init_plan_var(),
            backend.make_schedule(1), jax.random.PRNGKey(0),
        )


def _facade_with_context_backend():
    from genedynamics.solvers.single.mga.mga import MGASolver
    backend = _context_backend()
    nodes = backend.init_plan_var()
    backend.replan_with_info = lambda *args, **kw: (
        nodes, {"execution_mode": jnp.int32(1)}
    )
    solver = object.__new__(MGASolver)
    solver._backend_impl = backend
    solver._step_fn = lambda *_: pytest.fail("facade silently used ordinary transition")
    solver.config = {"Ndiffuse": 1, "Ndiffuse_init": 1}
    solver.seed, solver.n_steps, solver.method = 0, 1, "mga"
    solver._flatten_state = lambda state: np.asarray([state["steps"]], np.float32)
    return solver, backend


def test_context_direct_solve_autowires_real_execution_and_rejects_none_override():
    solver, backend = _facade_with_context_backend()
    result = solver.solve(_ContextToyTask.reset(), horizon=1)
    assert result.states[-1][0] == 1
    assert backend._committed_shift_mode == 1
    assert int(result.info["infos"][0]["execution_mode"]) == 1
    assert result.info["execution_mode_contract"] == {
        "schema_version": 1, "normal_mode": 0, "emergency_mode": 1,
    }
    for callback in (None, lambda state, action, info: state):
        with pytest.raises(ValueError, match="cannot be disabled or replaced"):
            solver.make_controller(1, execution_step=callback)
    # Rebinding the exact method yields a new Python object, not a new hook.
    assert backend._execute_with_context is not backend.execution_step
    rebound = solver.make_controller(1, execution_step=backend._execute_with_context)
    assert rebound._execution_step.__self__ is backend
    solver.make_controller(1, execution_step=backend.execution_step)


def test_contact_plugin_forwards_same_facade_execution_callback():
    from types import SimpleNamespace
    from genedynamics.experiments.plugins.methods.contact_receding import MGAMethodPlugin
    from genedynamics.solvers.common.receding_horizon import RecedingHorizonResult
    solver, backend = _facade_with_context_backend()
    def run_receding(initial, n_steps, rng, **kwargs):
        assert kwargs["execution_step"] is backend.execution_step
        return RecedingHorizonResult(
            states=[initial, initial], actions=[jnp.zeros(2)],
            infos=[{"execution_mode": jnp.int32(1)}],
        )
    solver.run_receding = run_receding
    env = SimpleNamespace(plan_initializer=None)
    planner = SimpleNamespace(
        solver=solver, env=env, execution_env=env, n_steps=1, seed=120,
        component_contract={},
    )
    output = MGAMethodPlugin().plan(planner, _ContextToyTask.reset(), None)
    assert int(output["infos"][0]["execution_mode"]) == 1


def test_context_acceptance_rejects_missing_emergency_candidate():
    backend = _context_backend()
    nodes = jnp.full((5, 2), 0.3)
    with pytest.raises(ValueError, match="explicit emergency candidate"):
        backend._accept_refinement(_ContextToyTask.reset(), nodes, nodes, jnp.float32(1))


class _GeometryContextToyTask(_ContextToyTask):
    """Distinct box geometry deliberately yields distinct candidate normals."""
    def __init__(self, half_size=(0.10, 0.10), radius=0.025, stiffness_scale=1.0):
        super().__init__(stiffness_scale=stiffness_scale)
        self.half_size = jnp.asarray(half_size, jnp.float32)
        self.radius = radius

    @staticmethod
    def reset(mode=0):
        return {
            **_ContextToyTask.reset(mode),
            "hand": jnp.asarray([0.12, 0.08], jnp.float32),
            "anchor": jnp.zeros(2, jnp.float32),
            "normal": jnp.zeros(2, jnp.float32),
        }

    def _capture_entry(self, state):
        hand = state["hand"]
        delta = hand - jnp.clip(hand, -self.half_size, self.half_size)
        distance = jnp.linalg.norm(delta)
        normal = delta / jnp.maximum(distance, jnp.float32(1e-12))
        target = hand + jnp.float32(0.02) * normal
        target_delta = target - jnp.clip(target, -self.half_size, self.half_size)
        target_gap = jnp.linalg.norm(target_delta) - self.radius
        geometry_valid = (
            (jnp.sum(delta != 0) == 1) & (target_gap > 0)
            & jnp.all(jnp.isfinite(target)) & jnp.all(jnp.isfinite(normal))
        )
        return {
            **super()._capture_entry(state),
            "anchor": target, "normal": normal,
            "entry_geometry_valid": geometry_valid,
        }


def _assert_same_sealed_entry(first, second):
    for key in ("anchor", "normal", "entry_prepared", "entry_geometry_valid",
                "stiffness_raw", "stiffness_matrix", "stiffness_valid"):
        np.testing.assert_array_equal(first[key], second[key])


def test_context_sealed_entry_is_idempotent_across_different_box_geometry():
    execution = _GeometryContextToyTask(half_size=(0.10, 0.10), radius=0.025)
    model = _GeometryContextToyTask(half_size=(0.20, 0.05), radius=0.005)
    measured = execution.reset()
    sealed = execution.prepare_state(measured, jnp.int32(1))
    alternative = model.prepare_state(measured, jnp.int32(1))
    assert not np.array_equal(sealed["anchor"], alternative["anchor"])
    assert bool(sealed["entry_prepared"]) and bool(sealed["entry_geometry_valid"])
    assert int(sealed["mode"]) == 0 and int(sealed["request"]) == 1
    for _ in range(3):
        repeated = model.prepare_state(sealed, jnp.int32(1))
        for key in sealed:
            np.testing.assert_array_equal(sealed[key], repeated[key])
        sealed = repeated
    predicted = model.step(sealed, jnp.zeros(2), jnp.int32(1))
    actual = execution.step(measured, jnp.zeros(2), jnp.int32(1))
    _assert_same_sealed_entry(sealed, predicted)
    _assert_same_sealed_entry(predicted, actual)
    assert int(predicted["mode"]) == int(actual["mode"]) == 1
    assert int(predicted["request"]) == int(actual["request"]) == -1


def test_context_invalid_but_prepared_entry_cannot_be_rescued_by_nominal_geometry():
    execution = _GeometryContextToyTask(radius=0.06)  # 2 cm retreat is insufficient
    model = _GeometryContextToyTask(half_size=(0.20, 0.05), radius=0.005)
    measured = execution.reset()
    sealed = execution.prepare_state(measured, jnp.int32(1))
    assert bool(sealed["entry_prepared"]) and not bool(sealed["entry_geometry_valid"])
    assert bool(model.prepare_state(measured, jnp.int32(1))["entry_geometry_valid"])
    repeated = model.prepare_state(sealed, jnp.int32(1))
    _assert_same_sealed_entry(sealed, repeated)
    _, risk = model.emergency_sequence_score_risk(sealed, jnp.zeros((17, 2)))
    assert bool(jnp.all(jnp.isinf(risk)))
    assert not bool(model.sequence_risk_is_safe(risk))


@pytest.mark.parametrize("committed,pending", [(0, 1), (1, -1)])
def test_context_missing_seal_is_unknown_not_permission_to_recapture(committed, pending):
    env = _GeometryContextToyTask()
    damaged = {
        **env.reset(), "mode": jnp.int32(committed), "request": jnp.int32(pending),
        "entry_prepared": jnp.asarray(False),
    }
    prepared = env.prepare_state(damaged, jnp.int32(1))
    assert not bool(prepared["entry_prepared"])
    assert not bool(prepared["entry_geometry_valid"])
    np.testing.assert_array_equal(prepared["anchor"], damaged["anchor"])
    _, risk = env.emergency_sequence_score_risk(prepared, jnp.zeros((17, 2)))
    assert bool(jnp.all(jnp.isinf(risk)))  # unknown certificate, not a force label


def test_context_normal_cancel_and_committed_recovery_have_distinct_entry_lifetimes():
    env = _GeometryContextToyTask()
    initial = env.reset()
    pending = env.prepare_state(initial, jnp.int32(1))
    cancelled = env.prepare_state(pending, jnp.int32(0))
    assert int(cancelled["mode"]) == 0 and int(cancelled["request"]) == 0
    assert not bool(cancelled["entry_prepared"])
    changed = {**cancelled, "hand": cancelled["hand"] + jnp.asarray([0.01, 0.0])}
    fresh = env.prepare_state(changed, jnp.int32(1))
    assert not np.array_equal(fresh["anchor"], pending["anchor"])
    committed = env.step(initial, jnp.zeros(2), jnp.int32(1))
    normal_trial = env.prepare_state(committed, jnp.int32(0))
    _assert_same_sealed_entry(committed, normal_trial)
    assert int(normal_trial["mode"]) == 1 and int(normal_trial["request"]) == 0
    continuing = env.prepare_state(normal_trial, jnp.int32(1))
    _assert_same_sealed_entry(committed, continuing)
    recovered = env.step(committed, jnp.zeros(2), jnp.int32(0))
    assert int(recovered["mode"]) == 0 and int(recovered["request"]) == -1
    assert not bool(recovered["entry_prepared"])


def test_context_backend_scoring_and_execution_share_seal_without_stale_plan_anchor():
    execution = _GeometryContextToyTask(half_size=(0.10, 0.10), radius=0.025)
    model = _GeometryContextToyTask(half_size=(0.20, 0.05), radius=0.005)
    backend = _context_backend(model=model, execution=execution)
    measured, nodes = execution.reset(), jnp.zeros((5, 2))
    emergency = backend.emergency_plan_fn(measured, nodes)
    selected, info = backend._accept_refinement(
        measured, nodes, nodes, jnp.float32(1), emergency=emergency,
    )
    assert int(info["execution_mode"]) == 1
    sealed = execution.prepare_state(measured, jnp.int32(1))
    expected_score, expected_risk = model.emergency_sequence_score_risk(
        sealed, backend.spline.node2u(emergency)
    )
    np.testing.assert_array_equal(info["prior_risk_emergency"], expected_risk)
    assert float(expected_score) == float(10.0 + jnp.sum(sealed["anchor"]))
    actual = backend.execution_step(measured, selected[0], info)
    _assert_same_sealed_entry(sealed, actual)

    # A deliberately changed input state must not consume an old speculative
    # entry. This checks reconstruction only: the old risk certificate is NOT
    # evidence for this changed state, which would require replanning.
    changed = {
        **sealed, "hand": measured["hand"] + jnp.asarray([0.01, 0.0], jnp.float32),
    }
    changed_actual = backend.execution_step(changed, selected[0], info)
    changed_fresh = execution.prepare_state(
        execution.prepare_state(changed, jnp.int32(0)), jnp.int32(1)
    )
    _assert_same_sealed_entry(changed_fresh, changed_actual)
    assert not np.array_equal(changed_actual["anchor"], sealed["anchor"])


def test_context_committed_unload_does_not_chase_a_changed_hand_or_box_model():
    execution = _GeometryContextToyTask()
    model = _GeometryContextToyTask(half_size=(0.20, 0.05), radius=0.005)
    committed = execution.step(execution.reset(), jnp.zeros(2), jnp.int32(1))
    changed = {
        **committed, "hand": committed["hand"] + jnp.asarray([0.03, -0.02], jnp.float32),
    }
    continued = model.step(changed, jnp.zeros(2), jnp.int32(1))
    _assert_same_sealed_entry(committed, continued)
    # Entry validity is historical. Real env code must separately validate
    # this unchanged target against current geometry/load at every substep.


def test_context_actual_callback_cannot_be_jitted_past_task_host_guards():
    backend = _context_backend()
    with pytest.raises(ValueError, match="must run on the host"):
        jax.jit(lambda state: backend.execution_step(
            state, jnp.zeros(2), {"execution_mode": jnp.int32(1)}
        ))(_ContextToyTask.reset())


def test_context_rejects_unrecoverable_emergency_before_physics():
    backend = _context_backend()
    state = _ContextToyTask.reset()
    with pytest.raises(ExecutionRejected, match="no_revalidated_safe_candidate") as error:
        backend.execution_step(
            state,
            jnp.zeros(2),
            {
                "execution_mode": jnp.int32(1),
                "emergency_unrecoverable": jnp.float32(1),
                "refined_revalidated_safe": jnp.float32(0),
                "incumbent_revalidated_safe": jnp.float32(0),
                "emergency_revalidated_safe": jnp.float32(0),
            },
        )
    assert error.value.details == {
        "normal_refined_safe": False,
        "incumbent_safe": False,
        "emergency_safe": False,
    }


def test_context_stiffness_seal_uses_last_action_cold_nominal_and_new_session():
    env = _GeometryContextToyTask(stiffness_scale=1.0)
    model = _GeometryContextToyTask(stiffness_scale=4.0)
    cold = env.prepare_state(env.reset(), jnp.int32(1))
    np.testing.assert_array_equal(cold["stiffness_raw"], np.zeros(6))
    np.testing.assert_array_equal(cold["stiffness_matrix"], np.eye(3))
    assert bool(cold["stiffness_valid"])

    measured = {**env.reset(), "prev_stiffness_raw": jnp.full(6, 0.2)}
    sealed = env.prepare_state(measured, jnp.int32(1))
    np.testing.assert_array_equal(sealed["stiffness_raw"], measured["prev_stiffness_raw"])
    alternative = model.prepare_state(measured, jnp.int32(1))
    assert not np.array_equal(sealed["stiffness_matrix"], alternative["stiffness_matrix"])
    predicted = model.step(sealed, jnp.zeros(2), jnp.int32(1))
    _assert_same_sealed_entry(sealed, predicted)
    np.testing.assert_array_equal(
        predicted["applied_stiffness_matrix"], sealed["stiffness_matrix"]
    )
    changed = {**sealed, "prev_stiffness_raw": jnp.full(6, -0.6)}
    repeated = env.prepare_state(changed, jnp.int32(1))
    _assert_same_sealed_entry(sealed, repeated)
    committed = env.step(changed, jnp.zeros(2), jnp.int32(1))
    _assert_same_sealed_entry(sealed, committed)
    np.testing.assert_array_equal(
        committed["applied_stiffness_matrix"], predicted["applied_stiffness_matrix"]
    )

    normal = env.step(committed, jnp.zeros(2), jnp.int32(0))
    assert not bool(normal["entry_prepared"]) and not bool(normal["stiffness_valid"])
    new_state = {**normal, "prev_stiffness_raw": jnp.full(6, -0.4)}
    new_entry = env.prepare_state(new_state, jnp.int32(1))
    np.testing.assert_array_equal(new_entry["stiffness_raw"], new_state["prev_stiffness_raw"])
    np.testing.assert_array_equal(
        new_entry["stiffness_matrix"], env._decode_stiffness(new_state["prev_stiffness_raw"])
    )


@pytest.mark.parametrize("bad_raw", [float("nan"), 1.1])
def test_context_invalid_entry_stiffness_is_sealed_unknown_not_silently_clipped(bad_raw):
    env = _GeometryContextToyTask()
    measured = {**env.reset(), "prev_stiffness_raw": jnp.full(6, bad_raw)}
    entry = env.prepare_state(measured, jnp.int32(1))
    assert bool(entry["entry_prepared"]) and not bool(entry["stiffness_valid"])
    changed = {**entry, "prev_stiffness_raw": jnp.zeros(6)}
    repeated = env.prepare_state(changed, jnp.int32(1))
    _assert_same_sealed_entry(entry, repeated)
    _, risk = env.emergency_sequence_score_risk(entry, jnp.zeros((17, 2)))
    assert bool(jnp.all(jnp.isinf(risk)))
