"""The strict guide reference advances through measured events in mode order."""
from dataclasses import replace
import numpy as np
import pytest
from types import SimpleNamespace

from genedynamics.tasks.transfer.controllers.bindings import (
    BINDING_VERSION, create_transfer_actor,
)
from genedynamics.tasks.transfer.controllers.guide_contact import (
    PublicGuideContactPolicy,
    PublicPredictiveGuideContactPolicy,
    PublicProgressQualifiedGuideContactPolicy,
    PublicWrenchGovernedGuideContactPolicy,
    PublicLowLoadWrenchGovernedGuideContactPolicy,
    PublicLowLoadPredictiveGuideContactPolicy,
    PublicAnisotropicPredictiveGuideContactPolicy,
    PublicSurfaceConsistentGuideContactPolicy,
    PublicChannelCenteredGuideContactPolicy,
    PublicSettledContactRampGuideContactPolicy,
    PublicLowLoadSettledGuideContactPolicy,
    PublicPostOutletLiftGuideContactPolicy,
    PublicDampedPostOutletLiftGuideContactPolicy,
    PublicBoundedUnloadingPostOutletLiftGuideContactPolicy,
    PublicFilteredNormalPostOutletLiftGuideContactPolicy,
    PublicHybridNormalPostOutletLiftGuideContactPolicy,
    PublicFiniteStateGuideContactMode0Policy,
    PublicFiniteStatePostOutletLiftMode0Policy,
    PublicFiniteStatePostOutletLeadLiftMode0Policy,
    PublicFiniteStatePostOutletPlanarLeadLiftMode0Policy,
    PublicFiniteStatePostOutletRawPlanarLeadLiftMode0Policy,
    PublicFiniteStatePostOutletConcurrentLiftMode0Policy,
    PublicFiniteStatePostOutletConcurrentLiftMode0PolicyV2,
    PublicFiniteStatePostOutletConcurrentLiftMode0PolicyV3,
    PublicDeadlineAwareGuideContactPolicy,
    PublicCompletionBudgetCaptureBandGuideCompletionPolicy,
    PublicCompletionBudgetCaptureGuideCompletionPolicy,
    PublicCompletionBudgetGuideCompletionPolicy,
    PublicCompletionBudgetPlanarBrakingGuideCompletionPolicy,
    PublicCompletionBudgetPreDescentSettlingGuideCompletionPolicy,
    PublicCompletionBudgetReferenceUnloadingGuideCompletionPolicy,
    PublicCompletionBudgetReferenceDebtUnloadingGuideCompletionPolicy,
    PublicCompletionBudgetCausalReferenceUnloadingGuideCompletionPolicy,
    PublicResponseNormalizedGuideCompletionPolicy,
    PublicTerminalOwnedGuideActor,
    public_finite_state_guide_contact_policy,
)
from genedynamics.tasks.transfer.contact import bounded_contact_command
from genedynamics.tasks.transfer.specs import task_spec


def observation(tick, position=(.38, -.20, .605), force=0., lateral=(0., 0.)):
    r = np.diag([1., -1., -1.])[None]
    return dict(
        elapsed_seconds=np.asarray([[tick * .02]]),
        tcp_position=np.asarray([position], float), tcp_rotation=r.copy(),
        reference_position=np.asarray([position], float), reference_rotation=r.copy(),
        goal_position=np.asarray([[.46, -.18, .596]], float),
        goal_rotation=r.copy(),
        tcp_twist=np.zeros((1, 6)), measured_wrench=np.asarray([[*lateral, force, 0., 0., 0.]]),
        previous_action=np.zeros((1, 13)),
    )


@pytest.mark.parametrize("mode", (0, 1))
def test_contact_hold_precedes_requested_axis_and_continuation_is_exact(mode):
    actor = PublicGuideContactPolicy(task_spec("T2"), mode=mode)
    for tick in range(12):
        public = observation(tick, force=0. if tick < 2 else 5.)
        before = actor.get_state()
        action = actor.act(public)
        actor.restore(before)
        np.testing.assert_array_equal(action, actor.act(public))
        np.testing.assert_array_equal(action, actor.act(public))
        if tick < 11:
            np.testing.assert_array_equal(action[0, :2], 0.)
            assert actor.stage < 2
    assert actor.stage == 2
    assert action[0, mode] > 0.
    assert action[0, 1 - mode] == 0.
    assert action[0, 2] >= 0.  # Surface feedforward with established support.
    public = observation(12, (.456, -.20, .595) if mode == 0 else (.38, -.1805, .591), force=5.)
    action = actor.act(public)
    assert actor.stage == 3 and actor.first_outlet_tick == 12
    assert action[0, 1 - mode] > 0.
    actor.act(observation(13, (.46, -.18, .596), force=5.))
    assert actor.stage == 4
    actor.reset()
    assert actor.stage == 0 and actor.first_contact_tick is None
    np.testing.assert_array_equal(actor.act(observation(0)), PublicGuideContactPolicy(task_spec("T2"), mode=mode).act(observation(0)))


def test_contact_loss_resets_hold_and_reference_does_not_lift_to_clear_walls():
    actor = PublicGuideContactPolicy(task_spec("T2"), mode=0)
    for tick in range(15):
        command = actor.act(observation(tick, force=5. if tick % 4 else 0.))
        assert actor.stage < 2
        np.testing.assert_array_equal(command[0, :2], 0.)
        assert command[0, 2] <= 0.
        assert command[0, 12] == 0.
    with pytest.raises(ValueError, match="every public"):
        actor.act(observation(17))
    state = actor.get_state()
    with pytest.raises(ValueError, match="binding"):
        PublicGuideContactPolicy(task_spec("T2"), mode=1).restore(state)


def test_finite_state_mode0_route_caps_relative_unloading_and_restores_exactly():
    spec = task_spec("T2")
    actor = PublicFiniteStateGuideContactMode0Policy(spec, mode=0)
    reference = PublicGuideContactPolicy(
        spec, mode=0, transit_force_N=2.0,
        wrench_governor=True, tangent_speed_limit_mps=.0075,
    )
    for tick in range(12):
        public = observation(tick, force=0. if tick < 2 else 5.)
        actor.act(public)
        reference.act(public)
    assert actor.stage == reference.stage == 2
    public = observation(12, position=(.441, -.20, .591), force=5.)
    before = actor.get_state()
    action = actor.act(public)
    nominal = reference.act(public)
    surface_velocity = .1 * float(action[0, 0] + action[0, 1])
    np.testing.assert_array_equal(action[0, [0, 1, 3, 4, 5, 12]], nominal[0, [0, 1, 3, 4, 5, 12]])
    assert action[0, 2] == min(
        float(nominal[0, 2]),
        surface_velocity + actor.MAXIMUM_RELATIVE_UNLOADING_MPS,
    )
    actor.restore(before)
    np.testing.assert_array_equal(actor.act(public), action)
    np.testing.assert_array_equal(actor.act(public), action)
    with pytest.raises(ValueError, match="requires mode 0"):
        PublicFiniteStateGuideContactMode0Policy(spec, mode=1)


def test_finite_state_mode0_speed_is_a_shared_task_setting():
    default_spec = task_spec("T2")
    slower_spec = replace(
        default_spec,
        config=replace(
            default_spec.config,
            guide_mode0_transit_speed_mps=.0065,
        ),
    )
    default = PublicFiniteStateGuideContactMode0Policy(default_spec, mode=0)
    slower = PublicFiniteStateGuideContactMode0Policy(slower_spec, mode=0)
    for tick in range(12):
        public = observation(tick, force=0. if tick < 2 else 5.)
        default_action = default.act(public)
        slower_action = slower.act(public)
    assert default.stage == slower.stage == 2
    assert default_action[0, 0] == pytest.approx(.0075)
    assert slower_action[0, 0] == pytest.approx(.0065)
    with pytest.raises(ValueError, match="deadline-reachable"):
        replace(default_spec.config, guide_mode0_transit_speed_mps=.006)


def test_deadline_reference_admission_is_task_owned_and_restores_exactly():
    fixed_spec = task_spec("T2")
    deadline_spec = replace(
        fixed_spec,
        config=replace(
            fixed_spec.config,
            guide_reference_admission="deadline_aware_v1",
        ),
    )
    fixed = public_finite_state_guide_contact_policy(fixed_spec, mode=0)
    admitted = public_finite_state_guide_contact_policy(deadline_spec, mode=0)
    assert isinstance(admitted, PublicDeadlineAwareGuideContactPolicy)
    for tick in range(11):
        public = observation(tick, force=0. if tick < 1 else 5.)
        fixed.act(public)
        admitted.act(public)
    public = observation(11, force=5.)
    before = admitted.get_state()
    fixed_action = fixed.act(public)
    admitted_action = admitted.act(public)
    assert admitted_action[0, 0] > fixed_action[0, 0]
    assert admitted_action[0, 1] == fixed_action[0, 1]
    assert admitted_action[0, 6:].tolist() == fixed_action[0, 6:].tolist()
    assert admitted.lead_limit_m > admitted.NOMINAL_LEAD_M
    assert admitted.admission_controls > 0
    admitted.restore(before)
    np.testing.assert_array_equal(admitted.act(public), admitted_action)

    # High measured utilization freezes the extra admission without changing
    # the fixed route, force target, impedance, or contact thresholds.
    public = observation(12, force=20.)
    np.testing.assert_array_equal(admitted.act(public), fixed.act(public))
    with pytest.raises(ValueError, match="unknown guide reference admission"):
        replace(fixed_spec.config, guide_reference_admission="private_oracle")


def test_mode0_post_outlet_lift_is_an_explicit_shared_task_setting():
    default_spec = task_spec("T2")
    lift_spec = replace(
        default_spec,
        config=replace(default_spec.config, guide_mode0_post_outlet_lift=True),
    )
    assert isinstance(
        public_finite_state_guide_contact_policy(default_spec, mode=0),
        PublicFiniteStateGuideContactMode0Policy,
    )
    actor = public_finite_state_guide_contact_policy(lift_spec, mode=0)
    assert isinstance(actor, PublicFiniteStatePostOutletLiftMode0Policy)
    for tick in range(12):
        actor.act(observation(tick, force=0. if tick < 2 else 5.))
    assert not actor.switched
    actor.act(observation(12, position=(.456, -.20, .596), force=5.))
    assert actor.switched and actor.switch_tick == 12
    with pytest.raises(TypeError, match="boolean"):
        replace(default_spec.config, guide_mode0_post_outlet_lift=1)


def test_mode0_lead_governed_lift_is_an_explicit_shared_task_setting():
    default_spec = task_spec("T2")
    lift_spec = replace(
        default_spec,
        config=replace(
            default_spec.config,
            guide_mode0_post_outlet_lead_governed_lift=True,
        ),
    )
    actor = public_finite_state_guide_contact_policy(lift_spec, mode=0)
    assert isinstance(actor, PublicFiniteStatePostOutletLeadLiftMode0Policy)
    with pytest.raises(ValueError, match="at most one"):
        replace(
            default_spec.config,
            guide_mode0_post_outlet_lift=True,
            guide_mode0_post_outlet_lead_governed_lift=True,
        )


def test_mode0_planar_lead_lift_uses_the_versioned_route_setting():
    default_spec = task_spec("T2")
    route_spec = replace(
        default_spec,
        config=replace(
            default_spec.config,
            guide_mode0_post_outlet_route="planar_lead_lift",
        ),
    )
    actor = public_finite_state_guide_contact_policy(route_spec, mode=0)
    assert isinstance(actor, PublicFiniteStatePostOutletPlanarLeadLiftMode0Policy)
    with pytest.raises(ValueError, match="cannot be combined"):
        replace(
            default_spec.config,
            guide_mode0_post_outlet_route="planar_lead_lift",
            guide_mode0_post_outlet_lift=True,
        )


def test_mode0_raw_planar_lead_lift_uses_the_versioned_route_setting():
    default_spec = task_spec("T2")
    route_spec = replace(
        default_spec,
        config=replace(
            default_spec.config,
            guide_mode0_post_outlet_route="raw_planar_lead_lift",
        ),
    )
    actor = public_finite_state_guide_contact_policy(route_spec, mode=0)
    assert isinstance(actor, PublicFiniteStatePostOutletRawPlanarLeadLiftMode0Policy)


def test_mode0_concurrent_lift_uses_the_versioned_route_setting():
    default_spec = task_spec("T2")
    route_spec = replace(
        default_spec,
        config=replace(
            default_spec.config,
            guide_mode0_post_outlet_route="concurrent_monotone_lift",
        ),
    )
    actor = public_finite_state_guide_contact_policy(route_spec, mode=0)
    assert isinstance(actor, PublicFiniteStatePostOutletConcurrentLiftMode0Policy)


def test_mode0_concurrent_lift_v2_uses_exact_task_tolerance_route():
    default_spec = task_spec("T2")
    route_spec = replace(
        default_spec,
        config=replace(
            default_spec.config,
            guide_mode0_post_outlet_route="concurrent_monotone_lift_v2",
        ),
    )
    actor = public_finite_state_guide_contact_policy(route_spec, mode=0)
    assert isinstance(actor, PublicFiniteStatePostOutletConcurrentLiftMode0PolicyV2)


def test_mode0_concurrent_lift_v3_restores_bounded_terminal_feedback():
    default_spec = task_spec("T2")
    route_spec = replace(
        default_spec,
        config=replace(
            default_spec.config,
            guide_mode0_post_outlet_route="concurrent_monotone_lift_v3",
        ),
    )
    actor = public_finite_state_guide_contact_policy(route_spec, mode=0)
    assert isinstance(actor, PublicFiniteStatePostOutletConcurrentLiftMode0PolicyV3)


def test_finite_state_binding_uses_the_two_frozen_public_routes():
    task = SimpleNamespace(spec=task_spec("T2"))
    common = {
        "version": BINDING_VERSION,
        "reference_version": "public_finite_state_guide_contact_v2",
    }
    mode_0 = create_transfer_actor(
        task, "public_geometry",
        {"mode": 0, "actor_binding": common},
    )
    mode_1 = create_transfer_actor(
        task, "public_geometry",
        {"mode": 1, "actor_binding": common},
    )
    assert isinstance(mode_0, PublicFiniteStateGuideContactMode0Policy)
    assert isinstance(mode_1, PublicPostOutletLiftGuideContactPolicy)


def test_low_transit_load_changes_only_loaded_motion_and_is_version_bound():
    spec = task_spec("T2")
    legacy = PublicGuideContactPolicy(spec, mode=0)
    revised = PublicGuideContactPolicy(spec, mode=0, transit_force_N=2.)
    for tick in range(12):
        public = observation(tick, force=0. if tick < 2 else 5.)
        old = legacy.act(public)
        new = revised.act(public)
    assert legacy.stage == revised.stage == 2
    np.testing.assert_array_equal(old[0, [0, 1, 3, 4, 5, 12]], new[0, [0, 1, 3, 4, 5, 12]])
    assert new[0, 2] > old[0, 2]
    assert legacy.get_state()["version"] != revised.get_state()["version"]
    with pytest.raises(ValueError, match="binding"):
        revised.restore(legacy.get_state())
    with pytest.raises(ValueError, match="contact load"):
        PublicGuideContactPolicy(spec, mode=0, transit_force_N=6.)


def test_public_wrench_governor_scales_only_tangent_and_keeps_normal_unloading():
    actor = PublicGuideContactPolicy(task_spec("T2"), mode=1, wrench_governor=True)
    actor.stage = 3
    assert actor._tangent_scale([0., 0., 0., 0., 0., 0.]) == 1.
    assert actor._tangent_scale([6., 0., 0., 0., 0., 0.]) == 0.
    assert actor._tangent_scale([0., 0., 0., 0., 0., 0.]) == .05
    for _ in range(20):
        recovered = actor._tangent_scale([0., 0., 0., 0., 0., 0.])
    assert 0. < recovered <= 1.
    actor.tick = -1
    low = actor.act(observation(0, force=5., lateral=(0., 0.)))
    actor.reset(); actor.stage = 3
    high = actor.act(observation(0, force=18., lateral=(6., 0.)))
    assert np.linalg.norm(low[0, :2]) > 0.
    np.testing.assert_array_equal(high[0, :2], 0.)
    assert high[0, 2] > 0.
    combined = PublicGuideContactPolicy(
        task_spec("T2"), mode=0, transit_force_N=2., wrench_governor=True,
        tangent_speed_limit_mps=.0075,
    )
    assert combined.version == combined.LOAD_RATE_WRENCH_GOVERNOR_VERSION
    rate = PublicGuideContactPolicy(
        task_spec("T2"), mode=0, wrench_governor=True,
        tangent_speed_limit_mps=.0075,
    )
    for tick in range(12):
        command = rate.act(observation(tick, force=0. if tick < 2 else 5.))
    assert rate.stage == 2
    assert 0. < command[0, 0] <= .0075
    with pytest.raises(ValueError, match="finite task reachability"):
        PublicGuideContactPolicy(
            task_spec("T2"), mode=0, wrench_governor=True,
            tangent_speed_limit_mps=.004,
        )
    filtered_normal = PublicGuideContactPolicy(
        task_spec("T2"), mode=0, wrench_governor=True,
        tangent_speed_limit_mps=.0075, normal_force_decay=.8,
    )
    filtered_normal.stage = 2
    filtered_normal.act(observation(0, force=10.))
    assert filtered_normal.normal_force_envelope == 10.
    filtered_normal.act(observation(1, force=0.))
    assert filtered_normal.normal_force_envelope == 8.
    with pytest.raises(ValueError, match="strict finite decay"):
        PublicGuideContactPolicy(
            task_spec("T2"), mode=0, wrench_governor=True,
            normal_force_decay=1.,
        )
    unilateral = PublicGuideContactPolicy(
        task_spec("T2"), mode=0, wrench_governor=True,
        tangent_speed_limit_mps=.0075, unilateral_transit=True,
    )
    unilateral.stage = 2
    command = unilateral.act(observation(0, force=0.))
    assert command[0, 2] >= 0.
    anisotropic = PublicGuideContactPolicy(
        task_spec("T2"), mode=0, wrench_governor=True,
        tangent_speed_limit_mps=(.0065, .01), unilateral_transit=True,
    )
    for tick in range(12):
        command = anisotropic.act(observation(tick, force=0. if tick < 2 else 5.))
    assert 0. < command[0, 0] <= .0065
    assert anisotropic.version == anisotropic.ANISOTROPIC_TRANSIT_GOVERNOR_VERSION


@pytest.mark.parametrize("mode", (0, 1))
def test_predictive_contact_path_stops_before_axis_switch_and_is_checkpointable(mode):
    actor = PublicPredictiveGuideContactPolicy(task_spec("T2"), mode=mode)
    previous = np.zeros((1, 13))
    for tick in range(12):
        public = observation(tick, force=0. if tick < 2 else 5.)
        public["previous_action"] = previous.copy()
        before = actor.get_state()
        action = actor.act(public)
        actor.restore(before)
        np.testing.assert_array_equal(action, actor.act(public))
        previous = action
    assert actor.stage == 2
    assert 0. < action[0, mode] <= actor.TRANSIT_SPEED_MPS
    assert abs(action[0, 1 - mode]) <= actor.CORRECTION_SPEED_MPS
    # Crossing the public outlet is insufficient while the measured tangent
    # speed is still high: the controller must stop before changing axes.
    position = (.456, -.20, .596) if mode == 0 else (.38, -.1805, .596)
    public = observation(12, position=position, force=5.)
    public["tcp_twist"][0, mode] = .006
    public["previous_action"] = previous.copy()
    actor.act(public)
    assert actor.stage == 2
    public = observation(13, position=(.46, -.20, .596) if mode == 0 else (.38, -.18, .596), force=5.)
    public["previous_action"] = actor.cached.copy()
    actor.act(public)
    assert actor.stage == 3 and actor.first_outlet_tick == 13


def test_predictive_projection_anticipates_nominal_tangent_wrench_and_freezes_version():
    actor = PublicPredictiveGuideContactPolicy(task_spec("T2"), mode=0)
    actor.stage = 2
    public = observation(0, position=(.39, -.20, .596), force=5.)
    public["reference_position"][0, :2] = [.397, -.20]
    public["previous_action"][0, :2] = [.012, 0.]
    action = actor.act(public)
    k = np.full(2, 800.)
    d = 2. * np.sqrt(k)
    lead = public["reference_position"][0, :2] - public["tcp_position"][0, :2]
    predicted = k * (lead + .02 * action[0, :2]) + d * (
        action[0, :2] - public["tcp_twist"][0, :2]
    )
    assert np.linalg.norm(predicted) <= actor.TANGENT_WRENCH_BUDGET_N + 1e-9
    assert actor.projection_active_ticks == 1
    assert actor.get_state()["version"] == actor.VERSION
    with pytest.raises(ValueError, match="binding"):
        PublicGuideContactPolicy(task_spec("T2"), mode=0).restore(actor.get_state())


def test_progress_qualified_version_changes_only_the_declared_projection_budget():
    conservative = PublicPredictiveGuideContactPolicy(task_spec("T2"), mode=0)
    progress = PublicProgressQualifiedGuideContactPolicy(task_spec("T2"), mode=0)
    assert progress.TANGENT_WRENCH_BUDGET_N == 9.0
    assert progress.VERSION != conservative.VERSION
    assert progress.TRANSIT_SPEED_MPS == conservative.TRANSIT_SPEED_MPS
    assert progress.ENDPOINT_TOLERANCE_M == conservative.ENDPOINT_TOLERANCE_M
    with pytest.raises(ValueError, match="binding"):
        progress.restore(conservative.get_state())


def test_wrench_governed_version_places_lead_limit_beyond_force_budget():
    actor = PublicWrenchGovernedGuideContactPolicy(task_spec("T2"), mode=0)
    assert 800.0 * actor.REFERENCE_LEAD_XY_M > actor.TANGENT_WRENCH_BUDGET_N
    assert actor.VERSION != PublicProgressQualifiedGuideContactPolicy.VERSION


def test_low_load_wrench_governor_reuses_qualified_traversal_regime():
    actor = PublicLowLoadWrenchGovernedGuideContactPolicy(
        task_spec("T2"), mode=0,
    )
    assert actor.TRANSIT_FORCE_N == 2.0
    assert actor.TANGENT_WRENCH_BUDGET_N == 12.0
    assert actor.TANGENT_STIFFNESS_SCALE == pytest.approx(np.e)
    assert actor.REFERENCE_LEAD_XY_M == 0.008
    assert actor.VERSION != PublicWrenchGovernedGuideContactPolicy.VERSION


def response_normalized_spec():
    spec = task_spec("T2")
    return replace(
        spec,
        config=replace(
            spec.config,
            guide_reference_admission="response_normalized_completion",
            log_stiffness_limit=1.5,
            tangential_wrench_limit=16.0,
        ),
    )


class _TerminalOwnershipCandidate:
    def __init__(self, action=None):
        self.calls = 0
        self.action = (
            np.full((1, 13), .25, dtype=np.float32)
            if action is None else np.asarray(action, dtype=np.float32).reshape(1, 13)
        )

    def act(self, public):
        self.calls += 1
        return self.action.copy()

    def reset(self):
        self.calls = 0

    def get_state(self):
        return {"calls": self.calls}

    def restore(self, state):
        self.calls = int(state["calls"])


class _TerminalOwnershipCommon:
    def __init__(self, terminal_tick=1, action=None):
        self.tick = -1
        self.terminal_tick = int(terminal_tick)
        self.switched = False
        self.terminal_pose_active = False
        self.force_utilization = 0.
        self.achieved_speed_mps = 0.
        self.clearance_lift_controls = 0
        self.action = (
            np.full((1, 13), -.5, dtype=np.float32)
            if action is None else np.asarray(action, dtype=np.float32).reshape(1, 13)
        )
        self.completion = SimpleNamespace(
            stage=0, initial=np.asarray([[.38, -.20, .605]], dtype=float),
        )

    def act(self, public):
        self.tick = int(round(float(public["elapsed_seconds"][0, 0]) / .02))
        self.switched = self.tick >= 1
        self.terminal_pose_active = self.tick >= self.terminal_tick
        return self.action.copy()

    def reset(self):
        self.tick = -1
        self.switched = False
        self.terminal_pose_active = False
        self.achieved_speed_mps = 0.

    def get_state(self):
        return {
            "tick": self.tick,
            "switched": self.switched,
            "terminal_pose_active": self.terminal_pose_active,
            "completion_stage": self.completion.stage,
            "clearance_lift_controls": self.clearance_lift_controls,
            "achieved_speed_mps": self.achieved_speed_mps,
        }

    def restore(self, state):
        self.tick = int(state["tick"])
        self.switched = bool(state["switched"])
        self.terminal_pose_active = bool(state["terminal_pose_active"])
        self.completion.stage = int(state["completion_stage"])
        self.clearance_lift_controls = int(
            state.get("clearance_lift_controls", 0)
        )
        self.achieved_speed_mps = float(
            state.get("achieved_speed_mps", 0.)
        )


def test_terminal_ownership_keeps_candidate_causal_but_selects_exact_common_action():
    candidate = _TerminalOwnershipCandidate()
    common = _TerminalOwnershipCommon()
    actor = PublicTerminalOwnedGuideActor(
        candidate, response_normalized_spec(), mode=1, common=common,
    )
    np.testing.assert_array_equal(
        actor.act(observation(0)), np.full((1, 13), .25, dtype=np.float32),
    )
    state = actor.get_state()
    expected = actor.act(observation(1))
    np.testing.assert_array_equal(
        expected, np.full((1, 13), -.5, dtype=np.float32),
    )
    assert candidate.calls == 2
    assert actor.terminal_ownership_active
    assert actor.terminal_ownership_controls == 1
    assert actor.terminal_ownership_activations == 1
    assert actor.last_selector_code == actor.COMMON_TERMINAL_SELECTED
    np.testing.assert_array_equal(actor.last_selected_action, actor.last_common_action)
    actor.restore(state)
    np.testing.assert_array_equal(actor.act(observation(1)), expected)


def test_post_outlet_ownership_starts_before_terminal_capture():
    candidate = _TerminalOwnershipCandidate()
    common = _TerminalOwnershipCommon(terminal_tick=2)
    actor = PublicTerminalOwnedGuideActor(
        candidate, response_normalized_spec(), mode=1,
        version=PublicTerminalOwnedGuideActor.POST_OUTLET_COMPLETION_VERSION,
        common=common,
    )
    np.testing.assert_array_equal(
        actor.act(observation(0)), np.full((1, 13), .25, dtype=np.float32),
    )
    assert not common.switched and not common.terminal_pose_active
    np.testing.assert_array_equal(
        actor.act(observation(1)), np.full((1, 13), -.5, dtype=np.float32),
    )
    assert common.switched and not common.terminal_pose_active
    assert actor.terminal_ownership_controls == 1


def test_clearance_progress_authority_blends_continuously_and_finishes_at_common():
    candidate = _TerminalOwnershipCandidate()
    common = _TerminalOwnershipCommon(terminal_tick=100)
    spec = response_normalized_spec()
    actor = PublicTerminalOwnedGuideActor(
        candidate, spec, mode=1,
        version=PublicTerminalOwnedGuideActor.CLEARANCE_PROGRESS_VERSION,
        common=common,
    )
    actor.act(observation(0))
    common.completion.stage = 1
    start = common.completion.initial[0, :2]
    goal = np.asarray(spec.config.goal[:2], dtype=float)
    midpoint = .5 * (start + goal)
    blended_public = observation(
        1, position=(midpoint[0], midpoint[1], .634),
    )
    blended = actor.act(blended_public)
    assert actor.candidate_authority_fraction == pytest.approx(.5)
    np.testing.assert_allclose(
        blended, np.full((1, 13), -.125, dtype=np.float32), atol=1e-7,
    )
    assert actor.last_selector_code == actor.CLEARANCE_BLEND_SELECTED
    common.completion.stage = 2
    terminal = actor.act(observation(
        2, position=(goal[0], goal[1], .634),
    ))
    np.testing.assert_array_equal(
        terminal, np.full((1, 13), -.5, dtype=np.float32),
    )
    assert actor.candidate_authority_fraction == 0.0


def test_terminal_set_admission_preserves_task_coordinates_and_goal_progress():
    candidate_action = np.zeros(13, dtype=np.float32)
    candidate_action[list(PublicTerminalOwnedGuideActor.TASK_OWNED_COORDINATES)] = 1.
    candidate_action[0] = -.01
    common_action = np.zeros(13, dtype=np.float32)
    candidate = _TerminalOwnershipCandidate(candidate_action)
    common = _TerminalOwnershipCommon(terminal_tick=100, action=common_action)
    spec = response_normalized_spec()
    actor = PublicTerminalOwnedGuideActor(
        candidate, spec, mode=1,
        version=PublicTerminalOwnedGuideActor.TERMINAL_SET_ADMISSION_VERSION,
        common=common,
    )
    common.completion.stage = 2
    public = observation(1, position=(.461, -.18, .620))
    selected = actor.act(public)
    assert candidate.calls == 1 and common.tick == 1
    assert actor.last_selector_code == actor.TERMINAL_PROGRESS_BLEND_SELECTED
    assert actor.candidate_authority_fraction == pytest.approx(1.)
    assert selected[0, 0] == pytest.approx(-.01)
    np.testing.assert_array_equal(
        selected[0, list(actor.TASK_OWNED_COORDINATES)],
        common_action[list(actor.TASK_OWNED_COORDINATES)],
    )
    assert actor.terminal_admission_controls == 1
    np.testing.assert_array_equal(actor.terminal_blend_counts, [1, 0, 0, 0])

    state = actor.get_state()
    common.force_utilization = .55
    expected = actor.act(observation(2, position=(.461, -.18, .620)))
    np.testing.assert_array_equal(expected, common_action[None])
    assert actor.terminal_common_fallback_controls == 1
    actor.restore(state)
    common.force_utilization = .55
    np.testing.assert_array_equal(
        actor.act(observation(2, position=(.461, -.18, .620))), expected,
    )


def test_terminal_set_admission_rejects_away_residual_and_stops_inside_set():
    candidate_action = np.zeros(13, dtype=np.float32)
    candidate_action[0] = -.01
    common_action = np.zeros(13, dtype=np.float32)
    actor = PublicTerminalOwnedGuideActor(
        _TerminalOwnershipCandidate(candidate_action),
        response_normalized_spec(), mode=1,
        version=PublicTerminalOwnedGuideActor.TERMINAL_SET_ADMISSION_VERSION,
        common=_TerminalOwnershipCommon(terminal_tick=100, action=common_action),
    )
    actor.common.completion.stage = 2
    away = actor.act(observation(1, position=(.459, -.18, .620)))
    np.testing.assert_array_equal(away, common_action[None])
    assert actor.last_selector_code == actor.COMMON_TERMINAL_SELECTED
    assert actor.terminal_common_fallback_controls == 1

    inside = actor.act(observation(2, position=(.46, -.18, .596)))
    np.testing.assert_array_equal(inside, common_action[None])
    assert actor.terminal_set_controls == 1
    assert actor.candidate.calls == 2


def test_terminal_transport_retains_force_safe_residual_through_dwell():
    candidate_action = np.zeros(13, dtype=np.float32)
    candidate_action[0] = -.01
    candidate_action[list(PublicTerminalOwnedGuideActor.TASK_OWNED_COORDINATES)] = 1.
    common_action = np.zeros(13, dtype=np.float32)
    actor = PublicTerminalOwnedGuideActor(
        _TerminalOwnershipCandidate(candidate_action),
        response_normalized_spec(), mode=1,
        version=PublicTerminalOwnedGuideActor.TERMINAL_TRANSPORT_VERSION,
        common=_TerminalOwnershipCommon(terminal_tick=100, action=common_action),
    )
    actor.common.completion.stage = 2

    away = actor.act(observation(1, position=(.459, -.18, .620)))
    assert away[0, 0] == pytest.approx(-.01)
    np.testing.assert_array_equal(
        away[0, list(actor.TASK_OWNED_COORDINATES)],
        common_action[list(actor.TASK_OWNED_COORDINATES)],
    )
    assert actor.last_selector_code == actor.TERMINAL_TRANSPORT_SELECTED
    assert actor.terminal_transport_controls == 1

    inside = actor.act(observation(2, position=(.46, -.18, .596)))
    assert inside[0, 0] == pytest.approx(-.01)
    assert actor.terminal_set_controls == 1
    assert actor.terminal_transport_controls == 2

    state = actor.get_state()
    actor.common.force_utilization = .55
    np.testing.assert_array_equal(
        actor.act(observation(3, position=(.46, -.18, .596))),
        common_action[None],
    )
    assert actor.terminal_force_fallback_controls == 1
    actor.restore(state)
    assert actor.terminal_force_fallback_controls == 0


def test_terminal_strict_progress_keeps_common_when_predictions_tie():
    candidate_action = np.zeros(13, dtype=np.float32)
    candidate_action[0] = .0095
    common_action = np.zeros(13, dtype=np.float32)
    common_action[0] = .01
    actor = PublicTerminalOwnedGuideActor(
        _TerminalOwnershipCandidate(candidate_action),
        response_normalized_spec(), mode=1,
        version=PublicTerminalOwnedGuideActor.TERMINAL_STRICT_PROGRESS_VERSION,
        common=_TerminalOwnershipCommon(terminal_tick=100, action=common_action),
    )
    actor.common.completion.stage = 2
    selected = actor.act(observation(1, position=(.4579, -.18, .596)))
    np.testing.assert_array_equal(selected, common_action[None])
    assert actor.last_selector_code == actor.COMMON_TERMINAL_SELECTED
    assert actor.terminal_common_fallback_controls == 1

    actor.reset()
    actor.common.completion.stage = 2
    actor.candidate.action[0] = .01
    actor.common.action[0] = 0.
    improving = actor.act(observation(1, position=(.455, -.18, .596)))
    assert improving[0, 0] == pytest.approx(.01)
    assert actor.last_selector_code == actor.TERMINAL_PROGRESS_BLEND_SELECTED


def test_handoff_viability_is_monotone_public_and_checkpointed():
    candidate_action = np.full(13, .02, dtype=np.float32)
    candidate_action[list(PublicTerminalOwnedGuideActor.TASK_OWNED_COORDINATES)] = 1.
    common_action = np.zeros(13, dtype=np.float32)
    common = _TerminalOwnershipCommon(terminal_tick=2000, action=common_action)
    actor = PublicTerminalOwnedGuideActor(
        _TerminalOwnershipCandidate(candidate_action),
        response_normalized_spec(), mode=1,
        version=PublicTerminalOwnedGuideActor.HANDOFF_VIABILITY_VERSION,
        common=common,
        completion_reserve_controls=150,
        reserve_taper_controls=100,
        reference_lag_full_authority_m=.008,
        reference_lag_zero_authority_m=.012,
        maximum_authority_decay_per_control=.05,
    )

    before_outlet = actor.act(observation(0))
    np.testing.assert_array_equal(before_outlet, candidate_action[None])

    lagged = observation(1)
    lagged["reference_position"][0, 0] += .012
    selected = actor.act(lagged)
    assert actor.handoff_authority_cap == pytest.approx(.95)
    assert actor.last_selector_code == actor.HANDOFF_VIABILITY_SELECTED
    np.testing.assert_allclose(
        selected[0, list(actor.TRANSPORT_COORDINATES)],
        .95 * candidate_action[list(actor.TRANSPORT_COORDINATES)],
    )
    np.testing.assert_array_equal(
        selected[0, list(actor.TASK_OWNED_COORDINATES)],
        common_action[list(actor.TASK_OWNED_COORDINATES)],
    )

    state = actor.get_state()
    recovered = actor.act(observation(2))
    assert actor.handoff_authority_cap == pytest.approx(.95)
    np.testing.assert_allclose(recovered, selected)
    actor.restore(state)
    np.testing.assert_array_equal(actor.act(observation(2)), recovered)

    common.completion.stage = 2
    terminal = actor.act(observation(3))
    np.testing.assert_array_equal(terminal, common_action[None])
    assert actor.handoff_authority_cap == 0.
    assert actor.handoff_exact_common_controls == 1
    assert actor.last_selector_code == actor.COMMON_TERMINAL_SELECTED

    mode0 = PublicTerminalOwnedGuideActor(
        _TerminalOwnershipCandidate(candidate_action),
        response_normalized_spec(), mode=0,
        version=PublicTerminalOwnedGuideActor.HANDOFF_VIABILITY_VERSION,
        common=_TerminalOwnershipCommon(action=common_action),
    )
    np.testing.assert_array_equal(mode0.act(observation(1)), candidate_action[None])


def test_handoff_viability_reserves_common_completion_before_deadline():
    common_action = np.zeros(13, dtype=np.float32)
    actor = PublicTerminalOwnedGuideActor(
        _TerminalOwnershipCandidate(np.full(13, .02, dtype=np.float32)),
        response_normalized_spec(), mode=1,
        version=PublicTerminalOwnedGuideActor.HANDOFF_VIABILITY_VERSION,
        common=_TerminalOwnershipCommon(terminal_tick=2000, action=common_action),
    )
    for tick in range(740, 851):
        selected = actor.act(observation(tick))
    assert actor.handoff_remaining_controls == 150
    assert actor.handoff_deadline_scale == 0.
    assert actor.handoff_authority_cap == 0.
    assert actor.last_selector_code == actor.COMMON_TERMINAL_SELECTED
    np.testing.assert_array_equal(selected, common_action[None])


def test_stage_debt_viability_releases_clearance_authority_monotonically():
    candidate_action = np.full(13, .02, dtype=np.float32)
    candidate_action[list(PublicTerminalOwnedGuideActor.TASK_OWNED_COORDINATES)] = 1.
    common_action = np.zeros(13, dtype=np.float32)
    common = _TerminalOwnershipCommon(terminal_tick=2000, action=common_action)
    actor = PublicTerminalOwnedGuideActor(
        _TerminalOwnershipCandidate(candidate_action),
        response_normalized_spec(), mode=1,
        version=PublicTerminalOwnedGuideActor.STAGE_DEBT_VIABILITY_VERSION,
        common=common,
        completion_reserve_controls=150,
        reserve_taper_controls=100,
        reference_lag_full_authority_m=.006,
        reference_lag_zero_authority_m=.010,
        maximum_authority_decay_per_control=.05,
        clearance_debt_full_controls=343,
        clearance_debt_zero_controls=371,
    )
    actor.act(observation(0))
    common.clearance_lift_controls = 343
    actor.act(observation(1))
    assert actor.handoff_clearance_debt_scale == 1.
    assert actor.handoff_authority_cap == 1.

    common.clearance_lift_controls = 357
    actor.act(observation(2))
    assert actor.handoff_clearance_debt_scale == pytest.approx(.5)
    assert actor.handoff_authority_cap == pytest.approx(.95)

    common.clearance_lift_controls = 371
    for tick in range(3, 22):
        selected = actor.act(observation(tick))
    assert actor.handoff_clearance_debt_scale == 0.
    assert actor.handoff_authority_cap == 0.
    assert actor.last_selector_code == actor.COMMON_TERMINAL_SELECTED
    np.testing.assert_array_equal(selected, common_action[None])
    np.testing.assert_array_equal(
        selected[0, list(actor.TASK_OWNED_COORDINATES)],
        common_action[list(actor.TASK_OWNED_COORDINATES)],
    )


def test_completion_budget_viability_is_stage1_bounded_and_checkpointed():
    candidate_action = np.full(13, .02, dtype=np.float32)
    candidate_action[list(PublicTerminalOwnedGuideActor.TASK_OWNED_COORDINATES)] = 1.
    common_action = np.zeros(13, dtype=np.float32)
    common = _TerminalOwnershipCommon(terminal_tick=2000, action=common_action)
    actor = PublicTerminalOwnedGuideActor(
        _TerminalOwnershipCandidate(candidate_action),
        response_normalized_spec(), mode=1,
        version=(
            PublicTerminalOwnedGuideActor.COMPLETION_BUDGET_VIABILITY_VERSION
        ),
        common=common,
        completion_progress_floor_mps=.020,
        post_traverse_reserve_controls=100,
        completion_budget_taper_controls=60,
        force_speed_full_authority_mps=.010,
        force_speed_zero_authority_mps=.025,
    )
    actor.act(observation(0))
    common.achieved_speed_mps = .030
    stage0 = actor.act(observation(1))
    assert actor.handoff_force_speed_scale == 1.
    assert actor.handoff_authority_cap == 1.
    np.testing.assert_array_equal(
        stage0[0, list(actor.TRANSPORT_COORDINATES)],
        candidate_action[list(actor.TRANSPORT_COORDINATES)],
    )
    np.testing.assert_array_equal(
        stage0[0, list(actor.TASK_OWNED_COORDINATES)],
        common_action[list(actor.TASK_OWNED_COORDINATES)],
    )

    common.completion.stage = 1
    for tick in range(700, 710):
        actor.act(observation(tick))
    state = actor.get_state()
    assert actor.handoff_completion_budget_controls > 0
    assert actor.handoff_completion_budget_scale < 1.
    assert actor.handoff_force_speed_scale == 0.
    authority = actor.handoff_authority_cap
    expected = actor.act(observation(710))
    assert actor.handoff_authority_cap < authority
    actor.restore(state)
    np.testing.assert_array_equal(actor.act(observation(710)), expected)

    for tick in range(711, 730):
        selected = actor.act(observation(tick))
    assert actor.handoff_authority_cap == 0.
    assert actor.last_selector_code == actor.COMMON_TERMINAL_SELECTED
    np.testing.assert_array_equal(selected, common_action[None])
    np.testing.assert_array_equal(
        selected[0, list(actor.TASK_OWNED_COORDINATES)],
        common_action[list(actor.TASK_OWNED_COORDINATES)],
    )


def test_response_normalization_is_public_bounded_and_checkpointed():
    spec = response_normalized_spec()
    actor = public_finite_state_guide_contact_policy(spec, mode=0)
    assert isinstance(actor, PublicResponseNormalizedGuideCompletionPolicy)
    public = observation(0)
    public["reference_position"][0, 0] += .004
    action = actor.act(public)
    assert actor.authority == pytest.approx(.02)
    assert 0. < action[0, 0] <= actor.MAXIMUM_FREE_SPEED_MPS
    assert actor.BASE_LOG_STIFFNESS < action[0, 6] <= 1.5
    bounded = bounded_contact_command(
        np, np.full(13, 10., dtype=np.float32), np.zeros(13), spec.config,
    )
    assert bounded[6] == pytest.approx(.08)
    previous = bounded.copy()
    for _ in range(20):
        bounded = bounded_contact_command(
            np, np.full(13, 10., dtype=np.float32), previous, spec.config,
        )
        previous = bounded.copy()
    assert bounded[6] == pytest.approx(1.5)

    state = actor.get_state()
    next_public = observation(1)
    next_public["reference_position"][0, 0] += .004
    expected = actor.act(next_public)
    restored = public_finite_state_guide_contact_policy(spec, mode=0)
    restored.restore(state)
    np.testing.assert_array_equal(restored.act(next_public), expected)
    restored_state = restored.get_state()
    expected_state = actor.get_state()
    np.testing.assert_array_equal(
        restored_state.pop("previous_position_m"),
        expected_state.pop("previous_position_m"),
    )
    for key in (
        "authority", "previous_axis", "achieved_speed_mps",
        "required_speed_mps", "tracking_lag_m", "force_utilization",
        "authority_controls", "progress_deficit_controls",
        "terminal_pose_active", "terminal_distance_m",
        "terminal_orientation_error_rad", "terminal_pose_hold_controls",
        "terminal_capture_controls", "terminal_reference_overshoot_m",
        "clearance_reference_settled", "clearance_settling_controls",
        "clearance_lift_controls",
        "normal_force_channel_controls",
    ):
        assert restored_state[key] == expected_state[key]


def test_response_normalization_releases_authority_on_measured_force():
    actor = public_finite_state_guide_contact_policy(
        response_normalized_spec(), mode=0,
    )
    actor.authority = .5
    public = observation(0, lateral=(6., 0.))
    public["reference_position"][0, 0] += .01
    actor.act(public)
    assert actor.force_utilization == pytest.approx(.6)
    assert actor.authority == pytest.approx(.46)
    with pytest.raises(ValueError, match="versioned stiffness support"):
        public_finite_state_guide_contact_policy(
            replace(
                task_spec("T2"),
                config=replace(
                    task_spec("T2").config,
                    guide_reference_admission="response_normalized_completion",
                    tangential_wrench_limit=16.,
                ),
            ),
            mode=0,
        )


def test_response_normalized_route_uses_bounded_terminal_admittance():
    actor = public_finite_state_guide_contact_policy(
        response_normalized_spec(), mode=0,
    )
    actor.route.stage = 3
    actor.route.first_outlet_tick = -50
    public = observation(0, position=(.46, -.18, .596), force=1.)
    action = actor.act(public)
    assert actor.route.stage == 4
    assert action[0, 12] == pytest.approx(
        actor.TERMINAL_SURFACE_FORCE_TARGET_N
    )
    assert action[0, 2] == 0.


def test_response_normalized_mode0_settles_first_axis_before_switching():
    actor = public_finite_state_guide_contact_policy(
        response_normalized_spec(), mode=0,
    )
    actor.route.stage = 2
    unsettled = observation(
        0, position=(.454, -.20, .596), force=5.,
    )
    unsettled["reference_position"][0, 0] = .461
    unsettled["tcp_twist"][0, 0] = .006
    action = actor.act(unsettled)
    assert actor.route.stage == 2
    assert actor.route.first_outlet_tick is None
    expected_limit = np.log(.55 * 16. / (800. * .007))
    assert actor.BASE_LOG_STIFFNESS < action[0, 6] <= expected_limit
    assert action[0, 1] == pytest.approx(0.)

    projected = observation(
        1, position=(.456, -.20, .596), force=5.,
    )
    projected["reference_position"][0, 0] = .462
    projected["tcp_twist"][0, 0] = .006
    action = actor.act(projected)
    assert actor.route.stage == 2
    assert actor.route.axis_endpoint_reached
    assert action[0, 0] < 0.
    expected_limit = np.log(.55 * 16. / (800. * .006))
    assert action[0, 6] <= expected_limit

    endpoint = observation(
        2, position=(.459, -.20, .596), force=5.,
    )
    endpoint["reference_position"][0, 0] = .467
    endpoint["tcp_twist"][0, 0] = .006
    action = actor.act(endpoint)
    assert actor.route.stage == 2
    assert actor.route.axis_endpoint_reached
    assert action[0, 0] < 0.

    settled = observation(
        3, position=(.459, -.20, .596), force=5.,
    )
    settled["reference_position"][0, 0] = .459
    settled["tcp_twist"][0, 0] = .003
    action = actor.act(settled)
    assert actor.route.stage == 3
    assert actor.route.first_outlet_tick == 3
    assert action[0, 1] > 0.


def test_response_normalized_mode0_outlet_latch_survives_reference_unloading():
    actor = public_finite_state_guide_contact_policy(
        response_normalized_spec(), mode=0,
    )
    actor.route.stage = 2
    crossed = observation(
        0, position=(.4551, -.20, .596), force=5.,
    )
    crossed["reference_position"][0, 0] = .463
    crossed["tcp_twist"][0, 0] = .006
    action = actor.act(crossed)
    assert actor.route.axis_endpoint_reached
    assert actor.route.stage == 2
    assert action[0, 0] < 0.

    unloaded = observation(
        1, position=(.4547, -.20, .596), force=5.,
    )
    unloaded["reference_position"][0, 0] = .4548
    unloaded["tcp_twist"][0, 0] = .001
    action = actor.act(unloaded)
    assert actor.route.stage == 3
    assert actor.route.first_outlet_tick == 1
    assert action[0, 1] > 0.


def test_response_normalized_surface_route_uses_public_hybrid_force_channel():
    actor = public_finite_state_guide_contact_policy(
        response_normalized_spec(), mode=0,
    )
    actor.route.stage = 1
    public = observation(0, position=(.398, -.20, .588), force=0.)
    public["reference_position"][0] = [.411, -.20, .581]
    action = actor.act(public)
    assert action[0, 2] == 0.
    assert action[0, 12] == pytest.approx(actor.spec.config.desired_force)
    assert actor.normal_force_channel_controls == 1
    assert actor.stage == actor.route.stage

    loaded = observation(1, position=(.398, -.20, .588), force=5.)
    loaded["reference_position"][0] = [.411, -.20, .581]
    action = actor.act(loaded)
    assert action[0, 2] == 0.
    assert action[0, 12] == pytest.approx(actor.spec.config.desired_force)
    assert actor.normal_force_channel_controls == 2

    far = public_finite_state_guide_contact_policy(
        response_normalized_spec(), mode=0,
    )
    far.route.stage = 0
    public = observation(0, position=(.38, -.20, .606), force=0.)
    public["reference_position"][0] = [.38, -.20, .606]
    action = far.act(public)
    assert action[0, 2] == 0.
    assert action[0, 12] == pytest.approx(far.spec.config.desired_force)


def test_response_normalized_mode1_switches_to_clearance_completion():
    actor = public_finite_state_guide_contact_policy(
        response_normalized_spec(), mode=1,
    )
    actor.route.stage = 3
    actor.route.first_outlet_tick = 0
    public = observation(0, position=(.38, -.18, .596), force=5.)
    action = actor.act(public)
    assert actor.switched and actor.completion_switch_tick == 0
    assert actor.completion.tick == 0
    np.testing.assert_array_equal(action[0, :2], 0.)
    assert action[0, 2] > 0.
    assert action[0, 2] <= actor.MAXIMUM_CONTACT_SPEED_MPS
    assert action[0, 8] == pytest.approx(1.5)
    assert action[0, 12] == 0.


def test_response_normalized_clearance_recenters_before_wrench_governed_lift():
    actor = public_finite_state_guide_contact_policy(
        response_normalized_spec(), mode=1,
    )
    actor.route.stage = 3
    actor.route.first_outlet_tick = 0
    public = observation(0, position=(.3807, -.181, .5881), force=2.)
    public["reference_position"][0, :2] = [.377, -.1787]
    action = actor.act(public)
    assert actor.switched and actor.completion.stage == 0
    assert not actor.clearance_reference_settled
    assert actor.clearance_settling_controls == 1
    assert action[0, 0] > 0.
    assert action[0, 1] < 0.
    assert action[0, 2] == 0.
    np.testing.assert_array_equal(
        action[0, 6:8], actor.BASE_LOG_STIFFNESS,
    )
    assert action[0, 8] == pytest.approx(1.5)

    actor.clearance_reference_settled = True
    actor.tick = 0
    actor.completion.tick = 0
    high = observation(1, position=(.3807, -.181, .5881), lateral=(5.5, 0.))
    high["reference_position"][0, :2] = [.3807, -.181]
    command = actor.act(high)
    assert actor.force_utilization == pytest.approx(.55)
    assert command[0, 2] == 0.


def test_response_normalized_terminal_capture_targets_the_set_boundary():
    actor = public_finite_state_guide_contact_policy(
        response_normalized_spec(), mode=1,
    )
    actor.route.stage = 3
    actor.route.first_outlet_tick = 0
    actor.switched = True
    actor.completion_switch_tick = 0
    actor.completion.stage = 2
    position = np.asarray([.4620900135, -.1804379172, .5961966952])
    reference = np.asarray([.4598513195, -.1786346573, .5930704333])
    public = observation(0, position=tuple(position), force=5.)
    public["reference_position"][0] = reference
    action = actor.act(public)

    expected_distance = max(
        0., np.linalg.norm(position - np.asarray(actor.spec.config.goal))
        - actor.spec.config.position_tolerance,
    )
    expected_seconds = (
        actor.spec.config.timeout_seconds
        - 2. * actor.TERMINAL_DWELL_CONTROLS * actor.spec.config.control_dt
    )
    assert actor.terminal_pose_active
    assert actor.terminal_distance_m == pytest.approx(expected_distance)
    assert actor.required_speed_mps == pytest.approx(
        expected_distance / expected_seconds,
    )
    assert actor.terminal_reference_overshoot_m == pytest.approx(
        min(actor.tracking_lag_m, actor.NOMINAL_LEAD_M),
    )
    assert actor.terminal_capture_controls == 1
    assert actor.terminal_pose_hold_controls == 0
    assert action[0, 0] < 0.
    assert abs(action[0, 0]) <= .010


def test_response_normalized_surface_completion_reserves_capture_time():
    actor = public_finite_state_guide_contact_policy(
        response_normalized_spec(), mode=0,
    )
    actor.route.stage = 3
    actor.route.first_outlet_tick = 0
    position = np.asarray([.458, -.184, .596])
    public = observation(0, position=tuple(position), force=5.)
    public["reference_position"][0] = position
    actor.act(public)
    remaining = max(
        0., np.linalg.norm(position - np.asarray(actor.spec.config.goal))
        - actor.spec.config.position_tolerance,
    )
    expected_seconds = (
        actor.spec.config.timeout_seconds
        - actor.COMPLETION_RESERVE_SECONDS
        - actor.TERMINAL_DWELL_CONTROLS * actor.spec.config.control_dt
    )
    assert actor.required_speed_mps == pytest.approx(
        remaining / expected_seconds,
    )


def test_response_normalized_clearance_traverse_brakes_into_descent_set():
    actor = public_finite_state_guide_contact_policy(
        response_normalized_spec(), mode=1,
    )
    actor.route.stage = 3
    actor.switched = True
    actor.completion_switch_tick = 0
    actor.completion.stage = 1
    goal = np.asarray(actor.spec.config.goal, dtype=float)
    position = goal + np.asarray([.006, 0., .038])
    public = observation(0, position=tuple(position), force=0.)
    public["reference_position"][0] = position
    public["reference_position"][0, 0] -= .002
    action = actor.act(public)
    remaining = np.linalg.norm(position[:2] - goal[:2]) - actor.spec.config.position_tolerance
    assert actor.completion.stage == 1
    assert actor.required_speed_mps == pytest.approx(remaining / 17.)
    assert actor.terminal_reference_overshoot_m > 0.
    assert action[0, 0] < 0.
    limit = max(actor.CLEARANCE_SETTLING_SPEED_MPS, 4. * remaining)
    assert abs(action[0, 0]) <= limit + 1e-7
    assert actor.tracking_lag_m + actor.spec.config.control_dt * abs(action[0, 0]) <= (
        actor.NOMINAL_LEAD_M
        + actor.authority * actor.ADDITIONAL_LEAD_M
        + 1e-7
    )


def test_completion_budget_liveness_raises_only_late_stage1_progress_request():
    legacy = PublicResponseNormalizedGuideCompletionPolicy(
        response_normalized_spec(), mode=1,
    )
    liveness = PublicCompletionBudgetGuideCompletionPolicy(
        response_normalized_spec(), mode=1,
    )
    actions = []
    for actor in (legacy, liveness):
        actor.route.stage = 3
        actor.route.first_outlet_tick = 0
        actor.switched = True
        actor.completion_switch_tick = 0
        actor.completion.stage = 1
        actor.tick = 839
        actor.completion.tick = 839
        public = observation(840, position=(.446, -.18, .634), force=0.)
        public["reference_position"][0] = public["tcp_position"][0]
        public["tcp_twist"][0, 0] = .082
        actions.append(actor.act(public))

    assert legacy.required_speed_mps == pytest.approx(.030)
    assert liveness.required_speed_mps == pytest.approx(.045)
    assert actions[0][0, 0] == pytest.approx(.030)
    assert actions[1][0, 0] == pytest.approx(.045)
    assert actions[1][0, 2] - actions[0][0, 2] == pytest.approx(.0015)
    np.testing.assert_array_equal(
        actions[0][0, [1, *range(3, 13)]],
        actions[1][0, [1, *range(3, 13)]],
    )


def test_completion_budget_capture_latches_v50_ceiling_after_first_set_entry():
    actor = PublicCompletionBudgetCaptureGuideCompletionPolicy(
        response_normalized_spec(), mode=1,
    )
    actor.route.stage = 3
    actor.route.first_outlet_tick = 0
    actor.switched = True
    actor.completion_switch_tick = 0
    actor.completion.stage = 1
    actor.tick = 889
    actor.completion.tick = 889

    far = observation(890, position=(.446, -.18, .634), force=0.)
    far["reference_position"][0] = far["tcp_position"][0]
    far["tcp_twist"][0, 0] = .082
    actor.act(far)
    assert not actor.clearance_capture_latched
    assert actor.required_speed_mps == pytest.approx(.045)

    inside = observation(891, position=(.459, -.18, .634), force=0.)
    inside["reference_position"][0] = inside["tcp_position"][0]
    inside["tcp_twist"][0, 0] = .05
    actor.act(inside)
    assert actor.completion.stage == 1
    assert actor.clearance_capture_latched

    outside = observation(892, position=(.446, -.18, .634), force=0.)
    outside["reference_position"][0] = outside["tcp_position"][0]
    outside["tcp_twist"][0, 0] = .082
    state = actor.get_state()
    actor.act(outside)
    assert actor.clearance_capture_latched
    assert actor.required_speed_mps == pytest.approx(.030)

    restored = PublicCompletionBudgetCaptureGuideCompletionPolicy(
        response_normalized_spec(), mode=1,
    )
    restored.restore(state)
    np.testing.assert_array_equal(restored.act(outside), actor.act(outside))
    assert restored.clearance_capture_latched


def test_completion_budget_capture_band_latches_before_planar_set_entry():
    actor = PublicCompletionBudgetCaptureBandGuideCompletionPolicy(
        response_normalized_spec(), mode=1,
    )
    actor.route.stage = 3
    actor.route.first_outlet_tick = 0
    actor.switched = True
    actor.completion_switch_tick = 0
    actor.completion.stage = 1
    actor.tick = 889
    actor.completion.tick = 889

    far = observation(890, position=(.446, -.18, .634), force=0.)
    far["reference_position"][0] = far["tcp_position"][0]
    far["tcp_twist"][0, 0] = .082
    actor.act(far)
    assert not actor.clearance_capture_latched
    assert actor.required_speed_mps == pytest.approx(.045)

    capture_band = observation(
        891, position=(.4485, -.18, .634), force=0.,
    )
    capture_band["reference_position"][0] = capture_band["tcp_position"][0]
    capture_band["tcp_twist"][0, 0] = .05
    actor.act(capture_band)
    assert actor.completion.stage == 1
    assert actor.clearance_capture_latched
    assert actor.required_speed_mps == pytest.approx(.030)
    state = actor.get_state()
    assert state["clearance_capture_band_m"] == pytest.approx(.010)

    outside = observation(892, position=(.446, -.18, .634), force=0.)
    outside["reference_position"][0] = outside["tcp_position"][0]
    outside["tcp_twist"][0, 0] = .082
    actor.act(outside)
    assert actor.required_speed_mps == pytest.approx(.030)


def test_completion_budget_planar_braking_owns_both_terminal_coordinates():
    legacy = PublicCompletionBudgetCaptureBandGuideCompletionPolicy(
        response_normalized_spec(), mode=1,
    )
    braking = PublicCompletionBudgetPlanarBrakingGuideCompletionPolicy(
        response_normalized_spec(), mode=1,
    )
    actions = []
    public = observation(
        860, position=(.466, -.174, .634), force=0.,
    )
    public["reference_position"][0] = public["tcp_position"][0]
    for actor in (legacy, braking):
        actor.route.stage = 3
        actor.route.first_outlet_tick = 500
        actor.switched = True
        actor.completion_switch_tick = 600
        actor.completion.stage = 2
        actor.tick = 859
        actor.completion.tick = 259
        actions.append(actor.act(public))

    assert np.linalg.norm(actions[0][0, :2]) > .010
    assert np.linalg.norm(actions[1][0, :2]) == pytest.approx(.010)
    room = (
        public["goal_position"][0, :2]
        - public["reference_position"][0, :2]
    ) / braking.spec.config.control_dt
    assert np.all(actions[1][0, :2] >= np.minimum(room, 0.))
    assert np.all(actions[1][0, :2] <= np.maximum(room, 0.))
    np.testing.assert_array_equal(actions[0][0, 2:], actions[1][0, 2:])
    assert braking.terminal_planar_speed_cap_active
    assert braking.terminal_planar_distance_m == pytest.approx(
        np.hypot(.006, .006) - braking.spec.config.position_tolerance
    )

    state = braking.get_state()
    assert state["terminal_planar_speed_cap_active"] is True
    restored = PublicCompletionBudgetPlanarBrakingGuideCompletionPolicy(
        response_normalized_spec(), mode=1,
    )
    restored.restore(state)
    assert restored.terminal_planar_speed_cap_active
    assert restored.terminal_planar_distance_m == pytest.approx(
        braking.terminal_planar_distance_m
    )


def test_completion_budget_planar_braking_preserves_stage1_exactly():
    legacy = PublicCompletionBudgetCaptureBandGuideCompletionPolicy(
        response_normalized_spec(), mode=1,
    )
    braking = PublicCompletionBudgetPlanarBrakingGuideCompletionPolicy(
        response_normalized_spec(), mode=1,
    )
    public = observation(840, position=(.446, -.18, .634), force=0.)
    public["reference_position"][0] = public["tcp_position"][0]
    public["tcp_twist"][0, 0] = .082
    actions = []
    for actor in (legacy, braking):
        actor.route.stage = 3
        actor.route.first_outlet_tick = 500
        actor.switched = True
        actor.completion_switch_tick = 600
        actor.completion.stage = 1
        actor.tick = 839
        actor.completion.tick = 239
        actions.append(actor.act(public))
    np.testing.assert_array_equal(actions[0], actions[1])
    assert not braking.terminal_planar_speed_cap_active
    assert braking.terminal_planar_distance_m == -1.


def test_completion_budget_pre_descent_settling_is_finite_and_checkpointed():
    legacy = PublicCompletionBudgetCaptureBandGuideCompletionPolicy(
        response_normalized_spec(), mode=1,
    )
    settling = PublicCompletionBudgetPreDescentSettlingGuideCompletionPolicy(
        response_normalized_spec(), mode=1,
    )
    for actor in (legacy, settling):
        actor.route.stage = 3
        actor.route.first_outlet_tick = 500
        actor.switched = True
        actor.completion_switch_tick = 600
        actor.completion.stage = 2
        actor.tick = 859
        actor.completion.tick = 259

    unsettled = observation(860, position=(.46, -.18, .634), force=0.)
    unsettled["reference_position"][0, :2] += [.008, 0.]
    unsettled["tcp_twist"][0, :2] = [.005, 0.]
    legacy.act(unsettled)
    held = settling.act(unsettled)
    np.testing.assert_array_equal(held[0, :3], 0.)
    assert settling.pre_descent_settling_active
    assert settling.pre_descent_settling_controls == 1
    assert settling.pre_descent_settled_controls == 0
    assert settling.pre_descent_reference_lead_norm_m == pytest.approx(.008)
    assert settling.pre_descent_planar_speed_mps == pytest.approx(.005)
    np.testing.assert_array_equal(settling.act(unsettled), held)
    assert settling.pre_descent_settling_controls == 1

    for tick in range(861, 864):
        public = observation(tick, position=(.46, -.18, .634), force=0.)
        public["reference_position"][0, :2] += [.001, 0.]
        public["tcp_twist"][0, :2] = [.003, 0.]
        legacy.act(public)
        np.testing.assert_array_equal(settling.act(public)[0, :3], 0.)
    assert settling.pre_descent_settled_controls == 3
    state = settling.get_state()
    restored = PublicCompletionBudgetPreDescentSettlingGuideCompletionPolicy(
        response_normalized_spec(), mode=1,
    )
    restored.restore(state)

    fourth = observation(864, position=(.46, -.18, .634), force=0.)
    fourth["reference_position"][0, :2] += [.001, 0.]
    fourth["tcp_twist"][0, :2] = [.003, 0.]
    legacy.act(fourth)
    action = settling.act(fourth)
    restored_action = restored.act(fourth)
    np.testing.assert_array_equal(action, restored_action)
    np.testing.assert_array_equal(action[0, :3], 0.)
    assert settling.pre_descent_settling_complete
    assert settling.pre_descent_settling_controls == 5
    assert settling.pre_descent_settled_controls == 4

    admitted = observation(865, position=(.46, -.18, .634), force=0.)
    admitted["reference_position"][0, :2] += [.001, 0.]
    admitted["tcp_twist"][0, :2] = [.003, 0.]
    legacy_action = legacy.act(admitted)
    np.testing.assert_array_equal(settling.act(admitted), legacy_action)
    np.testing.assert_array_equal(restored.act(admitted), legacy_action)
    assert not settling.pre_descent_settling_active


def test_completion_budget_pre_descent_settling_preserves_mode0_exactly():
    legacy = PublicCompletionBudgetCaptureBandGuideCompletionPolicy(
        response_normalized_spec(), mode=0,
    )
    settling = PublicCompletionBudgetPreDescentSettlingGuideCompletionPolicy(
        response_normalized_spec(), mode=0,
    )
    for tick in range(3):
        public = observation(tick, force=5.)
        np.testing.assert_array_equal(legacy.act(public), settling.act(public))
    assert not settling.pre_descent_settling_active
    assert not settling.pre_descent_settling_complete
    assert settling.pre_descent_settling_controls == 0


def test_completion_budget_reference_unloading_retracts_during_barrier():
    actor = PublicCompletionBudgetReferenceUnloadingGuideCompletionPolicy(
        response_normalized_spec(), mode=1,
    )
    actor.route.stage = 3
    actor.route.first_outlet_tick = 500
    actor.switched = True
    actor.completion_switch_tick = 600
    actor.completion.stage = 2
    actor.tick = 859
    actor.completion.tick = 259
    public = observation(860, position=(.46, -.18, .634), force=0.)
    public["reference_position"][0, :2] += [.008, -.004]
    public["tcp_twist"][0, :2] = [.005, 0.]
    action = actor.act(public)
    np.testing.assert_allclose(action[0, :3], [-.010, .010, 0.])
    assert actor.pre_descent_reference_unloading_active
    assert actor.pre_descent_reference_unloading_speed_mps == pytest.approx(
        np.sqrt(2.) * .010
    )
    assert actor.pre_descent_settling_controls == 1
    state = actor.get_state()
    restored = PublicCompletionBudgetReferenceUnloadingGuideCompletionPolicy(
        response_normalized_spec(), mode=1,
    )
    restored.restore(state)
    assert restored.pre_descent_reference_unloading_active
    assert restored.pre_descent_reference_unloading_speed_mps == pytest.approx(
        actor.pre_descent_reference_unloading_speed_mps
    )
    np.testing.assert_array_equal(restored.act(public), action)
    assert restored.pre_descent_settling_controls == 1


def test_completion_budget_reference_debt_unloading_uses_existing_limit():
    actor = PublicCompletionBudgetReferenceDebtUnloadingGuideCompletionPolicy(
        response_normalized_spec(), mode=1,
    )
    actor.route.stage = 3
    actor.route.first_outlet_tick = 500
    actor.switched = True
    actor.completion_switch_tick = 600
    actor.completion.stage = 2
    actor.tick = 859
    actor.completion.tick = 259
    public = observation(860, position=(.46, -.18, .634), force=0.)
    public["reference_position"][0, :2] += [.008, -.004]
    public["tcp_twist"][0, :2] = [.005, 0.]
    action = actor.act(public)
    np.testing.assert_allclose(action[0, :3], [-.030, .030, 0.])
    assert actor.PRE_DESCENT_REFERENCE_UNLOADING_LIMIT_MPS == (
        actor.MAXIMUM_CONTACT_SPEED_MPS
    )
    assert actor.pre_descent_reference_unloading_speed_mps == pytest.approx(
        np.sqrt(2.) * actor.MAXIMUM_CONTACT_SPEED_MPS
    )


def _identify_public_reference_delay(actor, delay):
    commands = np.asarray([
        [
            .003 * ((index % 5) - 2),
            .002 * (((2 * index) % 7) - 3),
        ]
        for index in range(16)
    ])
    reference = np.asarray([.38, -.20], dtype=float)
    buffer = [np.zeros(13, dtype=np.float32) for _ in range(4)]
    latest = None
    for tick in range(12):
        latest = observation(tick)
        latest.pop("previous_action")
        latest["reference_position"][0, :2] = reference
        actor.candidate.action.fill(0.)
        actor.candidate.action[0, :2] = commands[tick]
        actor.act(latest)
        bounded = actor.causal_last_bounded_action.copy()
        buffer = [bounded, *buffer[:-1]]
        reference += buffer[delay][:2] * .02
    return latest


@pytest.mark.parametrize("delay,limit", ((1, .030), (2, .015), (3, .010)))
def test_completion_budget_causal_reference_unloading_identifies_public_delay(
    delay, limit,
):
    actor = PublicTerminalOwnedGuideActor(
        _TerminalOwnershipCandidate(np.zeros(13)),
        response_normalized_spec(), mode=1,
        version=(
            PublicTerminalOwnedGuideActor.
            COMPLETION_BUDGET_CAUSAL_REFERENCE_UNLOADING_VERSION
        ),
    )
    latest = _identify_public_reference_delay(actor, delay)
    assert actor.public_reference_delay_estimate_controls == delay
    assert actor.public_reference_delay_identification_samples == 8
    assert actor.public_reference_delay_score_gap_mps > 0.
    assert actor.common.pre_descent_causal_unloading_limit_mps == pytest.approx(
        limit
    )

    # The observation is post-transition: delay d must address the inclusive
    # bounded-action history item -(d+1). Repeating that observation is exact.
    samples = actor.public_reference_delay_identification_samples
    repeated = actor.act(latest)
    np.testing.assert_array_equal(actor.act(latest), repeated)
    assert actor.public_reference_delay_identification_samples == samples

    restored = PublicTerminalOwnedGuideActor(
        _TerminalOwnershipCandidate(actor.candidate.action.copy()),
        response_normalized_spec(), mode=1,
        version=(
            PublicTerminalOwnedGuideActor.
            COMPLETION_BUDGET_CAUSAL_REFERENCE_UNLOADING_VERSION
        ),
    )
    restored.restore(actor.get_state())
    assert restored.public_reference_delay_estimate_controls == delay
    assert restored.public_reference_delay_identification_samples == 8
    assert restored.public_reference_delay_score_gap_mps == pytest.approx(
        actor.public_reference_delay_score_gap_mps
    )

    for wrapper in (actor, restored):
        policy = wrapper.common
        policy.route.stage = 3
        policy.route.first_outlet_tick = 5
        policy.switched = True
        policy.completion_switch_tick = 12
        policy.completion.stage = 2
    public = observation(12, position=(.46, -.18, .634), force=0.)
    public.pop("previous_action")
    public["reference_position"][0, :2] += [.008, -.004]
    public["tcp_twist"][0, :2] = [.005, 0.]
    action = actor.act(public)
    np.testing.assert_array_equal(restored.act(public), action)
    np.testing.assert_allclose(action[0, :3], [-limit, limit, 0.])
    assert actor.common.pre_descent_causal_unloading_limit_mps == pytest.approx(
        limit
    )


def test_completion_budget_causal_reference_unloading_falls_back_and_preserves_mode0():
    causal = PublicCompletionBudgetCausalReferenceUnloadingGuideCompletionPolicy(
        response_normalized_spec(), mode=0,
    )
    legacy = PublicCompletionBudgetReferenceUnloadingGuideCompletionPolicy(
        response_normalized_spec(), mode=0,
    )
    for tick in range(12):
        public = observation(tick, force=5. if tick > 1 else 0.)
        public["previous_action"][0, :2] = [.01, -.01]
        np.testing.assert_array_equal(causal.act(public), legacy.act(public))
    assert causal.public_reference_delay_estimate_controls == 0
    assert causal.public_reference_delay_identification_samples == 0
    assert causal.pre_descent_causal_unloading_limit_mps == pytest.approx(.010)


def test_completion_budget_causal_reference_unloading_fallback_is_v15_exact():
    legacy = PublicCompletionBudgetReferenceUnloadingGuideCompletionPolicy(
        response_normalized_spec(), mode=1,
    )
    causal = PublicCompletionBudgetCausalReferenceUnloadingGuideCompletionPolicy(
        response_normalized_spec(), mode=1,
    )
    for policy in (legacy, causal):
        policy.route.stage = 3
        policy.route.first_outlet_tick = 0
        policy.switched = True
        policy.completion_switch_tick = 0
        policy.completion.stage = 2
    public = observation(0, position=(.46, -.18, .634), force=0.)
    public["reference_position"][0, :2] += [.008, -.004]
    public["tcp_twist"][0, :2] = [.005, 0.]
    np.testing.assert_array_equal(causal.act(public), legacy.act(public))
    assert causal.public_reference_delay_estimate_controls == 0
    assert causal.public_reference_delay_identification_samples == 0
    assert causal.pre_descent_causal_unloading_limit_mps == pytest.approx(.010)

    legacy = PublicCompletionBudgetReferenceUnloadingGuideCompletionPolicy(
        response_normalized_spec(), mode=1,
    )
    causal = PublicCompletionBudgetCausalReferenceUnloadingGuideCompletionPolicy(
        response_normalized_spec(), mode=1,
    )
    for policy in (legacy, causal):
        policy.route.stage = 3
        policy.route.first_outlet_tick = 0
        policy.switched = True
        policy.completion_switch_tick = 0
        policy.completion.stage = 2
        policy.pre_descent_settling_complete = True
    np.testing.assert_array_equal(causal.act(public), legacy.act(public))


def test_response_normalized_clearance_descent_slows_outside_planar_set():
    actor = public_finite_state_guide_contact_policy(
        response_normalized_spec(), mode=1,
    )
    actor.route.stage = 3
    actor.switched = True
    actor.completion_switch_tick = 0
    actor.completion.stage = 2
    goal = np.asarray(actor.spec.config.goal, dtype=float)
    position = goal + np.asarray([.005, 0., .030])
    public = observation(0, position=tuple(position), force=0.)
    public["reference_position"][0] = position
    action = actor.act(public)
    assert actor.completion.stage == 2
    assert action[0, 2] >= -actor.MAXIMUM_CONTACT_SPEED_MPS

    high = public_finite_state_guide_contact_policy(
        response_normalized_spec(), mode=1,
    )
    high.route.stage = 3
    high.switched = True
    high.completion_switch_tick = 0
    high.completion.stage = 2
    loaded = observation(
        0, position=tuple(position), force=0., lateral=(5.5, 0.),
    )
    loaded["reference_position"][0] = position
    command = high.act(loaded)
    assert high.force_utilization == pytest.approx(.55)
    assert command[0, 2] >= 0.


def test_response_normalized_terminal_set_freezes_planar_reference_for_dwell():
    actor = public_finite_state_guide_contact_policy(
        response_normalized_spec(), mode=1,
    )
    actor.route.stage = 3
    actor.route.first_outlet_tick = 0
    actor.switched = True
    actor.completion_switch_tick = 0
    actor.completion.stage = 2
    goal = np.asarray(actor.spec.config.goal, dtype=float)
    position = goal + np.asarray([.001, 0., 0.])
    public = observation(0, position=tuple(position), force=5.)
    public["reference_position"][0] = goal - np.asarray([.001, .001, 0.])
    action = actor.act(public)
    np.testing.assert_array_equal(action[0, :2], 0.)
    assert actor.terminal_distance_m == 0.
    assert actor.terminal_pose_hold_controls == 1
    assert actor.terminal_capture_controls == 0
    assert actor.terminal_reference_overshoot_m == 0.

    state = actor.get_state()
    next_public = observation(1, position=tuple(position), force=5.)
    next_public["reference_position"][0] = public["reference_position"][0]
    expected = actor.act(next_public)
    restored = public_finite_state_guide_contact_policy(
        response_normalized_spec(), mode=1,
    )
    restored.restore(state)
    np.testing.assert_array_equal(restored.act(next_public), expected)
    assert restored.terminal_pose_hold_controls == 2


def test_response_normalized_terminal_wrench_retracts_public_reference_lead():
    actor = public_finite_state_guide_contact_policy(
        response_normalized_spec(), mode=1,
    )
    actor.route.stage = 3
    actor.switched = True
    actor.completion_switch_tick = 0
    actor.completion.stage = 2
    goal = np.asarray(actor.spec.config.goal, dtype=float)
    position = goal + np.asarray([.010, 0., .010])
    reference = position + np.asarray([.006, -.004, 0.])
    public = observation(
        0, position=tuple(position), lateral=(5.5, 0.),
    )
    public["reference_position"][0] = reference
    action = actor.act(public)
    assert actor.force_utilization == pytest.approx(.55)
    np.testing.assert_allclose(action[0, :2], [-.010, .010])
    assert actor.terminal_reference_overshoot_m == 0.


def test_low_load_predictive_version_tightens_only_tangent_budget():
    loose = PublicLowLoadWrenchGovernedGuideContactPolicy(
        task_spec("T2"), mode=0,
    )
    strict = PublicLowLoadPredictiveGuideContactPolicy(
        task_spec("T2"), mode=0,
    )
    assert strict.TANGENT_WRENCH_BUDGET_N == 9.0
    assert strict.TRANSIT_FORCE_N == loose.TRANSIT_FORCE_N
    assert strict.TANGENT_STIFFNESS_SCALE == loose.TANGENT_STIFFNESS_SCALE
    assert strict.VERSION != loose.VERSION


@pytest.mark.parametrize("mode", (0, 1))
def test_anisotropic_predictor_separates_transit_from_centering(mode):
    actor = PublicAnisotropicPredictiveGuideContactPolicy(
        task_spec("T2"), mode=mode,
    )
    actor.stage = 2
    scale = actor._tangent_stiffness_scale()
    lead = actor._tangent_lead_limits()
    assert scale[mode] == 1.0 and scale[1 - mode] == pytest.approx(np.e)
    assert lead[mode] == 0.012 and lead[1 - mode] == 0.008
    actor.stage = 3
    np.testing.assert_array_equal(actor._tangent_stiffness_scale(), scale[::-1])
    np.testing.assert_array_equal(actor._tangent_lead_limits(), lead[::-1])


def test_surface_consistent_version_adds_only_unilateral_normal_unloading():
    actor = PublicSurfaceConsistentGuideContactPolicy(
        task_spec("T2"), mode=0,
    )
    assert actor.SURFACE_LEAD_UNLOADING_GAIN == 4.0
    assert actor.TANGENT_WRENCH_BUDGET_N == 9.0
    assert actor.TRANSIT_FORCE_N == 2.0
    assert actor.VERSION != PublicAnisotropicPredictiveGuideContactPolicy.VERSION


def test_channel_centered_version_biases_only_mode0_first_corridor():
    mode0 = PublicChannelCenteredGuideContactPolicy(task_spec("T2"), mode=0)
    mode1 = PublicChannelCenteredGuideContactPolicy(task_spec("T2"), mode=1)
    goal = np.asarray(task_spec("T2").config.goal)
    mode0.stage = mode1.stage = 2
    assert mode0._target_for_stage(goal)[1] == -0.202
    assert mode1._target_for_stage(goal)[0] == 0.38
    mode0.stage = 3
    np.testing.assert_array_equal(mode0._target_for_stage(goal)[:2], goal[:2])
    assert mode0.TRANSIT_FORCE_N == 2.0
    assert mode0.TANGENT_STIFFNESS_SCALE == 1.0


def test_settled_contact_ramp_is_stage_clocked_and_checkpointed():
    actor = PublicSettledContactRampGuideContactPolicy(
        task_spec("T2"), mode=0,
    )
    actor.stage = 2
    actor.stage_entry_tick = 20
    actor.tick = 19
    assert actor._tangent_wrench_budget() == 4.0
    actor.tick = 69
    assert actor._tangent_wrench_budget() == pytest.approx(6.5)
    actor.tick = 119
    assert actor._tangent_wrench_budget() == 9.0
    restored = PublicSettledContactRampGuideContactPolicy(
        task_spec("T2"), mode=0,
    )
    restored.restore(actor.get_state())
    assert restored.stage_entry_tick == 20


def test_low_load_settled_version_changes_only_traversal_force():
    actor = PublicLowLoadSettledGuideContactPolicy(task_spec("T2"), mode=0)
    assert actor.TRANSIT_FORCE_N == 1.0
    assert actor.TANGENT_WRENCH_BUDGET_N == 9.0
    assert actor.TANGENT_WRENCH_RAMP_SECONDS == 2.0
    assert actor.VERSION != PublicSettledContactRampGuideContactPolicy.VERSION


@pytest.mark.parametrize("mode", (0, 1))
def test_post_outlet_route_switch_is_public_exact_and_checkpointed(mode):
    actor = PublicPostOutletLiftGuideContactPolicy(task_spec("T2"), mode=mode)
    first = actor.act(observation(0, force=5.0))
    assert not actor.switched
    position = (.456, -.20, .596) if mode == 0 else (.38, -.1805, .596)
    public = observation(1, position=position, force=5.0)
    public["previous_action"] = first.copy()
    state = actor.get_state()
    switched = actor.act(public)
    assert actor.switched and actor.switch_tick == 1
    actor.restore(state)
    np.testing.assert_array_equal(actor.act(public), switched)
    assert switched[0, 2] > 0.0


def test_damped_outlet_version_changes_only_normal_admittance():
    actor = PublicDampedPostOutletLiftGuideContactPolicy(
        task_spec("T2"), mode=0,
    )
    assert actor.base.NORMAL_ADMITTANCE_GAIN == 0.001
    assert actor.base.TANGENT_WRENCH_BUDGET_N == 9.0
    assert actor.base.TRANSIT_FORCE_N == 0.75
    assert actor.VERSION != PublicPostOutletLiftGuideContactPolicy.VERSION


def test_bounded_unloading_version_caps_only_upward_contact_velocity():
    actor = PublicBoundedUnloadingPostOutletLiftGuideContactPolicy(
        task_spec("T2"), mode=0,
    )
    assert actor.base.NORMAL_ADMITTANCE_GAIN == 0.001
    assert actor.base.NORMAL_UNLOADING_LIMIT_MPS == 0.0015
    assert actor.base.TRANSIT_FORCE_N == 0.75


def test_filtered_normal_version_is_checkpointed_and_keeps_original_gain():
    actor = PublicFilteredNormalPostOutletLiftGuideContactPolicy(
        task_spec("T2"), mode=0,
    )
    assert actor.base.NORMAL_FORCE_FILTER_DECAY == 0.9
    assert actor.base.NORMAL_ADMITTANCE_GAIN == 0.004
    actor.base.filtered_normal_force = 1.25
    state = actor.get_state()
    actor.base.filtered_normal_force = 0.0
    actor.restore(state)
    assert actor.base.filtered_normal_force == 1.25


def test_hybrid_normal_version_uses_existing_force_channel_only_after_settle():
    actor = PublicHybridNormalPostOutletLiftGuideContactPolicy(
        task_spec("T2"), mode=0,
    )
    actor.base.stage = 2
    public = observation(0, force=5.0)
    action = actor.act(public)
    assert action[0, 12] == pytest.approx(0.2)  # Existing 10 N/s slew.
    assert actor.base.TRANSIT_HYBRID_FORCE_N == 0.75


