"""Backend-neutral contact task and hybrid controller contracts.

Positive command[12] is compressive normal force in N. The normal is world +z;
positive force pushes down. command[0:6] remains world spatial twist. Force
feedback is measured from constraints, never copied from the commanded force.
"""
from dataclasses import dataclass
import math
from typing import Any,NamedTuple
from .runtime import MechanismSpec


GOVERNOR_NOMINAL = 0
GOVERNOR_BRAKE = 1
GOVERNOR_RECONTACT = 2
GOVERNOR_RELEASE = 3
GOVERNOR_BRAKE_SUBSTEPS = 10
GOVERNOR_SUPPORT_DWELL_SUBSTEPS = 4
GOVERNOR_RELEASE_SUBSTEPS = 10
GOVERNOR_SOFT_FRACTION = .8
GOVERNOR_BRAKE_TANGENT_N = 2.5
GOVERNOR_SEPARATION_DAMPING_NS_PER_M = 1000.
GOVERNOR_SEPARATION_BRAKE_N = 1.
GOVERNOR_CONTACT_CAPTURE_GAP_M = .0001
GOVERNOR_SEPARATION_BRAKE_RAMP_M = .0001
SUBSTEP_WRENCH_EXECUTION_PROFILES = frozenset({
    "substep_wrench_governor",
    "substep_wrench_governor_recontact_reference",
    "substep_wrench_governor_settling_reference",
    "substep_wrench_governor_recontact_settling",
    "substep_wrench_governor_separation_edge",
})


@dataclass(frozen=True)
class ContactConfig:
    environment: str="guided_contact_alignment"
    robot: str="panda"
    physics_dt: float=.002
    control_dt: float=.02
    timeout_seconds: float=20.
    force_limit: float=25.
    desired_force: float=5.
    force_rate: float=10.
    contact_loss_seconds: float=.3
    force_integral_limit: float=10.
    contact_threshold: float=.3
    reset_joint_noise: float=0.
    reset_task_noise: float=0.
    insertion_depth: float=.046
    position_tolerance: float=.008
    orientation_tolerance: float=math.pi
    orientation_wrench_limit: float=2.
    lateral_force_limit: float=25.
    approach_speed: float=.005
    approach_integral_limit: float=.01
    mobile: bool=False
    socket_profile: str="legacy"
    mating_clearance: float=.0008
    wrench_semantics: str="legacy_aggregate"
    guide_profile: str="legacy"
    force_feedback_gain: float=.05
    force_integral_gain: float=.05
    tangential_wrench_limit: float=12.
    log_stiffness_limit: float=1.
    execution_profile: str="legacy"
    reset_profile: str="legacy"
    guide_mode0_transit_speed_mps: float=.0075
    guide_mode0_post_outlet_route: str="contact"
    guide_mode0_post_outlet_lift: bool=False
    guide_mode0_post_outlet_lead_governed_lift: bool=False
    guide_reference_admission: str="fixed_lead_v1"

    def __post_init__(self):
        if self.execution_profile not in {
            "legacy", "joint_margin", "joint_margin_bounds",
            "joint_margin_public_reference", *SUBSTEP_WRENCH_EXECUTION_PROFILES,
        }:
            raise ValueError("unknown contact execution profile")
        if (self.execution_profile in {
                "joint_margin", "joint_margin_bounds",
                "joint_margin_public_reference",
        } and self.robot != "go2_z1"):
            raise ValueError("joint-margin execution requires the floating Go2+Z1 WBC")
        if (self.execution_profile in SUBSTEP_WRENCH_EXECUTION_PROFILES
                and self.guide_profile != "spatial_dual_channel"):
            raise ValueError("substep wrench governor requires the spatial guide task")
        if self.reset_profile not in {"legacy", "folded_stance"}:
            raise ValueError("unknown contact reset profile")
        if self.reset_profile == "folded_stance" and self.robot != "go2_z1":
            raise ValueError("folded stance preparation requires Go2+Z1")
        if self.socket_profile not in {"legacy","matched_keyway"}: raise ValueError("unknown socket profile")
        if not math.isfinite(self.mating_clearance) or not 0<self.mating_clearance<=.002:
            raise ValueError("mating clearance outside declared 0..2 mm pilot support")
        if self.wrench_semantics not in {"legacy_aggregate","tool_at_tcp_world"}: raise ValueError("unknown wrench semantics")
        if self.guide_profile not in {"legacy","spatial_dual_channel","planar_corner_v1"}: raise ValueError("unknown guide profile")
        if self.guide_profile!="legacy" and self.insertion:raise ValueError("spatial guide requires guided contact alignment")
        if self.socket_profile!="legacy" and not self.insertion:
            raise NotImplementedError("matched keyway requires insertion")
        if self.environment not in {
                "guided_contact_alignment", "keyed_insertion_recovery",
                "mobile_keyed_insertion_recovery", "reposition_then_dock",
        }:
            raise ValueError("unknown contact environment")
        if self.robot not in {"panda","xarm7","ur5e","go2_z1"}: raise ValueError("unsupported contact arm")
        if any(not math.isfinite(x) or x<=0 for x in (self.physics_dt,self.control_dt,self.timeout_seconds,
             self.force_limit,self.desired_force,self.force_rate,self.contact_loss_seconds,self.force_integral_limit)):
            raise ValueError("contact limits/clocks must be finite positive")
        if any(not math.isfinite(v) or v<0 for v in (self.force_feedback_gain,self.force_integral_gain)):
            raise ValueError("force feedback gains must be finite nonnegative")
        if not math.isfinite(self.tangential_wrench_limit) or self.tangential_wrench_limit<=0:raise ValueError("invalid tangential wrench limit")
        if (not math.isfinite(self.log_stiffness_limit)
                or not 1. <= self.log_stiffness_limit <= 2.):
            raise ValueError("log stiffness limit must stay in the declared contact range")
        if (not math.isfinite(self.guide_mode0_transit_speed_mps)
                or not .00613 <= self.guide_mode0_transit_speed_mps <= .0075):
            raise ValueError("mode-0 guide speed must preserve the frozen deadline-reachable range")
        if not isinstance(self.guide_mode0_post_outlet_lift, bool):
            raise TypeError("mode-0 post-outlet lift must be boolean")
        if not isinstance(
            self.guide_mode0_post_outlet_lead_governed_lift, bool
        ):
            raise TypeError("mode-0 post-outlet lead lift must be boolean")
        if (self.guide_mode0_post_outlet_lift
                and self.guide_mode0_post_outlet_lead_governed_lift):
            raise ValueError("select at most one mode-0 post-outlet route")
        if self.guide_mode0_post_outlet_route not in {
            "contact", "planar_lead_lift", "raw_planar_lead_lift",
            "concurrent_monotone_lift", "concurrent_monotone_lift_v2",
            "concurrent_monotone_lift_v3",
        }:
            raise ValueError("unknown mode-0 post-outlet route")
        if (self.guide_mode0_post_outlet_route != "contact"
                and (self.guide_mode0_post_outlet_lift
                     or self.guide_mode0_post_outlet_lead_governed_lift)):
            raise ValueError("legacy and versioned mode-0 routes cannot be combined")
        if self.guide_reference_admission not in {
            "fixed_lead_v1", "deadline_aware_v1",
            "response_normalized_completion",
        }:
            raise ValueError("unknown guide reference admission")
        if not math.isclose(self.control_dt/self.physics_dt,round(self.control_dt/self.physics_dt)):
            raise ValueError("control clock must contain integer physics steps")
        if self.desired_force>=self.force_limit: raise ValueError("desired force must be below failure force")
        if not math.isfinite(self.reset_joint_noise) or self.reset_joint_noise<0: raise ValueError("invalid contact reset noise")
        if not math.isfinite(self.reset_task_noise) or self.reset_task_noise<0: raise ValueError("invalid task reset noise")
        if any(not math.isfinite(x) or x<=0 for x in (self.insertion_depth,self.position_tolerance,self.orientation_tolerance,self.orientation_wrench_limit,self.lateral_force_limit,self.approach_speed,self.approach_integral_limit)):
            raise ValueError("invalid geometry or acceptance limits")
        if self.insertion_depth>.046 or self.orientation_tolerance>math.pi: raise ValueError("unsupported geometry or orientation tolerance")
        if self.robot=="go2_z1" and self.reset_task_noise: raise NotImplementedError("floating task-space noisy reset is not qualified")
        if not math.isfinite(self.contact_threshold) or not 0<self.contact_threshold<self.desired_force: raise ValueError("invalid contact threshold")
        if self.robot=="go2_z1" and not self.mobile: raise ValueError("Go2+Z1 requires a mobile body configuration")

    @property
    def insertion(self): return self.environment!="guided_contact_alignment"
    @property
    def substeps(self): return round(self.control_dt/self.physics_dt)
    @property
    def timeout_steps(self): return math.ceil(self.timeout_seconds/self.control_dt)
    @property
    def goal(self):
        if self.insertion:
            return (.44,-.20,.596-self.insertion_depth)
        if self.guide_profile=="spatial_dual_channel":
            return (.46,-.18,.596)
        if self.guide_profile=="planar_corner_v1":
            return (.44,-.136,.586)
        return (.46,-.18,.586)


def research_contact_config(robot="panda"):
    """Frozen CPU insertion protocol: 16 mm depth, 14 mm minimum, 2 deg, 15 s.

    Legacy defaults remain available for historical functional qualification.
    The common angular limit accommodates known native joint stiction.
    """
    return ContactConfig(environment="keyed_insertion_recovery",robot=robot,reset_task_noise=.002,
        insertion_depth=.016,position_tolerance=.002,orientation_tolerance=math.radians(2),
        orientation_wrench_limit=4.,lateral_force_limit=10.,timeout_seconds=15.,approach_speed=.015,approach_integral_limit=.03)


def mating_contact_config(robot="panda"):
    """Development pilot with a physically matching keyway, not final test data."""
    from dataclasses import replace
    return replace(research_contact_config(robot),socket_profile="matched_keyway",
                   wrench_semantics="tool_at_tcp_world",reset_task_noise=.0005)


def spatial_guide_config(robot="panda"):
    """Two narrow contact routes over a rising 3D surface, around a real ridge."""
    return ContactConfig(environment="guided_contact_alignment",robot=robot,guide_profile="spatial_dual_channel",
        wrench_semantics="tool_at_tcp_world",position_tolerance=.002,orientation_tolerance=math.radians(2),
        reset_task_noise=.0003,orientation_wrench_limit=4.,lateral_force_limit=10.,timeout_seconds=20.,
        approach_speed=.005,approach_integral_limit=.01,force_feedback_gain=.5,force_integral_gain=.5)


def corner_guide_config(robot="xarm7"):
    """Right-angle planar guide with solid walls and a physical end stop."""
    return ContactConfig(
        environment="guided_contact_alignment", robot=robot,
        guide_profile="planar_corner_v1", wrench_semantics="tool_at_tcp_world",
        desired_force=4.,
        position_tolerance=.002, orientation_tolerance=math.radians(2),
        reset_task_noise=.0002, orientation_wrench_limit=4.,
        lateral_force_limit=10., timeout_seconds=40., approach_speed=.005,
        approach_integral_limit=.01, force_feedback_gain=.5,
        force_integral_gain=.5,
    )


@dataclass(frozen=True)
class ContactMechanismSpec(MechanismSpec):
    """Construction-only contact deviations; these are never policy inputs."""
    contact_friction: float=.4
    contact_time_constant: float=.02

    def __post_init__(self):
        super().__post_init__()
        if any(not math.isfinite(v) or v<=0 for v in (self.contact_friction,self.contact_time_constant)):
            raise ValueError("contact mechanism must be finite positive")


@dataclass(frozen=True)
class MobileContactMechanismSpec(ContactMechanismSpec):
    """Construction-only independent support and arm actuator responses.

    The shared arm command delay does not delay the support controller. Foot
    friction remains the known nominal model in this first stance protocol.
    """
    body_servo_gain: float=1.

    def __post_init__(self):
        super().__post_init__()
        if not math.isfinite(self.body_servo_gain) or not 0<self.body_servo_gain<=2.:
            raise ValueError("invalid support servo gain")


class ContactState(NamedTuple):
    physics: Any
    reference_position: Any
    reference_rotation: Any
    last_command: Any
    command_buffer: Any
    force_integral: Any
    approach_integral: Any
    body_reference: Any
    measured_force: Any
    governor_phase: Any
    governor_phase_age: Any
    governor_previous_wrench: Any
    governor_support_dwell: Any
    governor_release_progress: Any
    sample_id: Any
    steps: Any
    contact_seen: Any
    loss_count: Any
    hold_count: Any
    status: Any


class ContactTrace(NamedTuple):
    tcp_position: Any
    tcp_rotation: Any
    command: Any
    measured_wrench: Any
    normal_force: Any
    force_peak: Any
    allowed_contact: Any
    forbidden_collision: Any
    contact_lost: Any
    saturation: Any
    force_integral: Any
    stage: Any  # 0 approach,1 loaded,2 withdrawn,3 inserted
    status: Any
    valid: Any
    terminal: Any
    lateral_force_peak: Any  # max norm of summed world-xy contact force over physics substeps


class ContactChunk(NamedTuple):
    state: ContactState
    trace: ContactTrace


class ContactCommands(NamedTuple):
    skill: Any       # [B,L,13], unchanged shared task skill
    body_twist: Any  # [B,L,3], world planar vx/vy/yaw rate, low-level body intent


def bounded_contact_command(xp,command,previous,config):
    amplitude=xp.asarray(
        [.08]*3+[.4]*3+[config.log_stiffness_limit]*6+[config.force_limit],
        dtype=command.dtype,
    )
    rates=xp.asarray([.4]*3+[2.]*3+[4.]*6+[config.force_rate],dtype=command.dtype)*config.control_dt
    value=previous+xp.clip(xp.clip(command,-amplitude,amplitude)-previous,-rates,rates)
    return xp.concatenate((value[:12],xp.maximum(value[12:13],0.)))


def hybrid_wrench(xp,position,rotation,twist,reference,reference_rotation,command,integral,measured,config,contact_seen=False,approach_integral=0.):
    from genedynamics.core.control.transfer import orientation_error
    stiffness=xp.asarray([800.]*3+[60.]*3,dtype=position.dtype)*xp.exp(command[6:12])
    error=xp.concatenate((reference-position,orientation_error(xp,rotation,reference_rotation)))
    wrench=stiffness*error+2*xp.sqrt(stiffness)*(command[:6]-twist)
    loaded=command[12]>.01
    # Position projection (I - nn^T) in the force-controlled normal channel.
    force=-command[12]-config.force_feedback_gain*(command[12]-measured)-config.force_integral_gain*integral-80.*twist[2]
    # Bounded admittance approach before establishing contact avoids applying
    # the full load in free air and striking a stiff fixture at high speed.
    force=xp.where(contact_seen|(measured>config.contact_threshold),force,-.4+800.*approach_integral-80.*twist[2])
    normal=xp.where(loaded,force,wrench[2])
    return xp.concatenate((xp.clip(wrench[:2],-config.tangential_wrench_limit,config.tangential_wrench_limit),normal[None],xp.clip(wrench[3:],-config.orientation_wrench_limit,config.orientation_wrench_limit)))


def govern_substep_reference(
    xp, reference, position, twist, command, applied_wrench, phase, config,
    *, support_dwell=None,
):
    """Admit a brake reference inside full-effort and energy budgets.

    The 50 Hz actor keeps its original reference-integration semantics.  When
    the existing finite brake is active, this 2 ms projection bounds the full
    tangent impedance effort (spring plus damping) after reserving the applied
    normal effort.  It separately bounds stored spring effort, then writes the
    admitted reference back to shared state.  Nominal, recontact, and other
    execution profiles return the reference bit-for-bit.
    """
    enabled = config.execution_profile in SUBSTEP_WRENCH_EXECUTION_PROFILES
    dtype = reference.dtype
    candidate = reference
    stiffness = xp.asarray([800., 800.], dtype=dtype) * xp.exp(command[6:8])
    spring = stiffness * (candidate[:2] - position[:2])
    damping = 2. * xp.sqrt(stiffness) * (command[:2] - twist[:2])
    project_recontact = (
        config.execution_profile
        == "substep_wrench_governor_recontact_reference"
    )
    project_settling = (
        config.execution_profile
        == "substep_wrench_governor_settling_reference"
    )
    if project_settling and support_dwell is None:
        raise ValueError("settling reference projection requires support dwell")
    dwell = (
        GOVERNOR_SUPPORT_DWELL_SUBSTEPS
        if support_dwell is None else support_dwell
    )
    settling = (
        project_settling
        & (phase == GOVERNOR_RECONTACT)
        & (dwell < GOVERNOR_SUPPORT_DWELL_SUBSTEPS)
    )
    brake = enabled & (
        (phase == GOVERNOR_BRAKE)
        | (project_recontact & (phase == GOVERNOR_RECONTACT))
        | settling
    )
    total_soft = xp.asarray(
        GOVERNOR_SOFT_FRACTION * config.force_limit, dtype=dtype,
    )
    lateral_soft = xp.asarray(
        GOVERNOR_SOFT_FRACTION * config.lateral_force_limit, dtype=dtype,
    )
    normal_reserve = xp.minimum(xp.abs(applied_wrench[2]), total_soft)
    total_tangent_allowance = xp.sqrt(xp.maximum(
        total_soft * total_soft - normal_reserve * normal_reserve,
        xp.asarray(0., dtype=dtype),
    ))
    effort_allowance = xp.minimum(lateral_soft, total_tangent_allowance)

    # Construct a point in the intersection
    #   ||spring + damping|| <= effort_allowance
    #   ||spring|| <= lateral_soft.
    # First cancel only the portion of damping outside the effort ball.  The
    # remaining spring increment is limited by both residual radii, so the
    # triangle inequality proves both bounds without a backend-specific solve.
    eps = xp.asarray(1e-12, dtype=dtype)
    damping_norm = xp.linalg.norm(damping)
    cancel_magnitude = xp.maximum(damping_norm - effort_allowance, 0.)
    cancellation = -damping * cancel_magnitude / xp.maximum(damping_norm, eps)
    stored_residual = xp.maximum(lateral_soft - cancel_magnitude, 0.)
    effort_residual = xp.maximum(
        effort_allowance - xp.minimum(damping_norm, effort_allowance), 0.,
    )
    residual_limit = xp.minimum(stored_residual, effort_residual)
    requested_residual = spring - cancellation
    residual_norm = xp.linalg.norm(requested_residual)
    admitted_residual = requested_residual * xp.minimum(
        1., residual_limit / xp.maximum(residual_norm, eps),
    )
    projected_spring = cancellation + admitted_residual
    projected_effort = projected_spring + damping
    projected = xp.concatenate((
        position[:2] + projected_spring / stiffness,
        candidate[2:3],
    ))
    energy = .5 * xp.sum(projected_spring * projected_spring / stiffness)
    governed = xp.where(brake, projected, candidate)
    changed = xp.linalg.norm(governed[:2] - reference[:2]) > 0.
    zero = xp.asarray(0., dtype=dtype)
    admitted_effort = xp.where(brake, projected_effort, applied_wrench[:2])
    return (governed, xp.where(brake, energy, zero), brake & changed,
            xp.where(brake, effort_allowance, zero),
            xp.where(brake, xp.linalg.norm(projected_effort), zero),
            xp.where(brake, xp.linalg.norm(projected_spring), zero),
            admitted_effort)


def govern_substep_wrench(
    xp, desired, issued_tangent, measured, position, twist,
    measured_support, established, phase, phase_age, previous_measured,
    support_dwell, release_progress, config,
):
    """Checkpointable 2 ms guide-contact recovery state machine.

    The public actor remains a 50 Hz skill policy.  This pure numerical guard
    runs inside the already-declared low-level contact controller and uses only
    its measured TCP wrench/twist and public contact-memory state.  Contact
    geometry, evaluator peaks, mechanism parameters and mode labels are not
    inputs.

    The recorded T2 impacts can jump from zero to the hard limit in one
    physics step, so a wrench derivative alone cannot intervene in time.  A
    separation edge or measured wrench-growth risk enters a finite brake,
    recontact, and release sequence.  Every phase and its measured history is
    part of ContactState, so repeated loss samples cannot restart a local
    one-tick timer.  Outside the state machine the desired wrench is returned
    bit-for-bit.  Contact identity and evaluator axial-support truth are not
    inputs.
    """
    enabled = config.execution_profile in SUBSTEP_WRENCH_EXECUTION_PROFILES
    force = xp.linalg.norm(measured[:3])
    previous_force = xp.linalg.norm(previous_measured[:3])
    lateral = xp.linalg.norm(measured[:2])
    previous_lateral = xp.linalg.norm(previous_measured[:2])
    predicted_force = force + xp.maximum(force - previous_force, 0.)
    predicted_lateral = lateral + xp.maximum(lateral - previous_lateral, 0.)
    total_soft = GOVERNOR_SOFT_FRACTION * config.force_limit
    lateral_soft = GOVERNOR_SOFT_FRACTION * config.lateral_force_limit
    budget_risk = ((predicted_force >= total_soft)
                   | (predicted_lateral >= lateral_soft))
    forward_x = ((issued_tangent[0] >= .006)
                 & (issued_tangent[0] >= xp.abs(issued_tangent[1])))
    forward_y = ((issued_tangent[1] >= .006)
                 & (issued_tangent[1] >= xp.abs(issued_tangent[0])))
    separation_edge = (established & forward_x
                       & (previous_force > 1.) & (force < .3))
    entrance_soft = position[0] <= .390
    precontact_transit = (established & forward_x & entrance_soft
                          & ~measured_support
                          & (xp.linalg.norm(desired[:2]) >= 8.))
    surface_height = (.586 + .1 * (position[0] - .38)
                      + .1 * (position[1] + .20))
    surface_gap = position[2] - surface_height
    relative_normal_velocity = twist[2] - .1 * (twist[0] + twist[1])
    low_risk_support = (measured_support & (force <= .5 * total_soft)
                        & (lateral <= .5 * lateral_soft))
    support_dwell = xp.where(low_risk_support, support_dwell + 1, 0)
    brake_substeps = round(config.control_dt / config.physics_dt)
    support_dwell_substeps = max(1, round(.008 / config.physics_dt))
    separation_edge_brake = (
        config.execution_profile
        == "substep_wrench_governor_separation_edge"
    )
    tangent_trigger = budget_risk | (separation_edge_brake & separation_edge)
    route_channel = xp.where(
        (release_progress == 0) & established & forward_y,
        2,
        xp.where(
            (release_progress == 0) & established & forward_x,
            1,
            release_progress,
        ),
    )
    contact_route = established & (
        (route_channel != 0) | forward_x | forward_y
    )
    entering_brake = (
        ((phase == GOVERNOR_NOMINAL) | (phase == GOVERNOR_RECONTACT))
        & tangent_trigger
    )
    phase = xp.where(
        entering_brake,
        GOVERNOR_BRAKE,
        phase,
    )
    phase = xp.where(
        (phase == GOVERNOR_NOMINAL) & contact_route,
        GOVERNOR_RECONTACT,
        phase,
    )
    reset_brake_age = separation_edge_brake & entering_brake
    phase_age = xp.where(
        reset_brake_age,
        1,
        xp.where(phase == GOVERNOR_NOMINAL, 0, phase_age + 1),
    )
    brake_release = ((phase == GOVERNOR_BRAKE)
                     & ((support_dwell >= support_dwell_substeps)
                        | (phase_age >= brake_substeps)))
    phase = xp.where(
        brake_release,
        xp.where(contact_route, GOVERNOR_RECONTACT, GOVERNOR_NOMINAL),
        phase,
    )
    phase = xp.where(
        (phase == GOVERNOR_RECONTACT) & ~established,
        GOVERNOR_NOMINAL,
        phase,
    )
    phase_age = xp.where(phase == GOVERNOR_NOMINAL, 0, phase_age)
    release_progress = xp.where(
        phase == GOVERNOR_NOMINAL,
        xp.asarray(0, dtype=phase_age.dtype),
        route_channel,
    )

    def cap_tangent(value, limit):
        norm = xp.linalg.norm(value)
        return value * xp.minimum(1., limit / xp.maximum(norm, 1e-12))

    brake_scale = xp.where(precontact_transit, .5, .1)
    brake_tangent = cap_tangent(
        brake_scale * desired[:2] - 120. * twist[:2],
        xp.where(precontact_transit, 6., GOVERNOR_BRAKE_TANGENT_N),
    )
    tangent = brake_tangent
    y_target = xp.asarray(1., dtype=desired.dtype)
    # A lost-contact rebound can have a small geometric gap but enough
    # outward velocity to remain airborne longer than the strict 150 ms
    # support window.  Brake only that outward motion.  Once the tool turns
    # back toward the guide this term is exactly zero, so it does not add
    # reimpact energy.
    separation_gap_scale = xp.clip(
        (surface_gap - GOVERNOR_CONTACT_CAPTURE_GAP_M)
        / GOVERNOR_SEPARATION_BRAKE_RAMP_M,
        0., 1.,
    )
    separation_brake = separation_gap_scale * xp.clip(
        GOVERNOR_SEPARATION_DAMPING_NS_PER_M
        * xp.maximum(relative_normal_velocity, 0.),
        0., GOVERNOR_SEPARATION_BRAKE_N,
    )
    y_recovery_normal = xp.clip(
        -y_target - .25 * (y_target - measured[2])
        - 400. * surface_gap - 80. * relative_normal_velocity
        - separation_brake,
        -3., 3.,
    )
    finite_recontact_settling = (
        config.execution_profile
        == "substep_wrench_governor_recontact_settling"
    )
    x_route = (release_progress == 1) | forward_x
    settling_x_route = (
        finite_recontact_settling
        & x_route
        & (support_dwell < support_dwell_substeps)
    )
    settled_support = measured_support & ~settling_x_route
    y_normal = xp.where(settled_support, desired[2],
                        xp.minimum(desired[2], y_recovery_normal))
    y_route_memory = (release_progress == 2)
    x_governed = xp.concatenate((
        tangent, y_normal[None], desired[3:],
    ))
    recontact_tangent = xp.where(
        settled_support | y_route_memory,
        desired[:2],
        brake_tangent,
    )
    y_governed = xp.concatenate((
        recontact_tangent, y_normal[None], desired[3:],
    ))
    x_active = enabled & (phase == GOVERNOR_BRAKE)
    recontact_active = enabled & (phase == GOVERNOR_RECONTACT)
    active = x_active | recontact_active
    desired = xp.where(x_active, x_governed,
                       xp.where(recontact_active, y_governed, desired))
    return (desired, phase, phase_age, measured, support_dwell,
            release_progress, active)


def contact_termination(xp,state,position,normal,peak,forbidden,finite,saturated,config,*,rotation=None,lateral_peak=0.,axial_contact=None,seated_history=None):
    seen=xp.where(state.last_command[12]>.1,state.contact_seen|(normal>config.contact_threshold),False)
    lost=seen&(state.last_command[12]>.1)&(normal<config.contact_threshold)
    loss_count=xp.where(lost,state.loss_count+1,0)
    if config.socket_profile=="matched_keyway" or config.guide_profile in {"spatial_dual_channel","planar_corner_v1"}:
        if axial_contact is None: raise ValueError("matched keyway requires an axial support witness")
        # Brushing a vertical wall is not established axial support. During
        # insertion such contacts can be intermittent; continuous support is
        # expected only near the seat. Force/collision limits still apply.
        seen=(state.last_command[12]>.1)&axial_contact&(normal>config.contact_threshold)
        near_seat=position[2]<config.goal[2]+config.position_tolerance if config.insertion else True
        history=state.contact_seen|(state.loss_count>0) if seated_history is None else seated_history
        lost=history&(state.last_command[12]>.1)&~seen&near_seat
        loss_count=xp.where(lost,state.loss_count+1,0)
    reached=(xp.linalg.norm(position-xp.asarray(config.goal))<config.position_tolerance)&(xp.abs(normal-config.desired_force)<2.5)
    if rotation is not None:
        from genedynamics.core.control.transfer import orientation_distance
        reached=reached&(orientation_distance(xp,rotation,xp.diag(xp.asarray([1.,-1.,-1.])))<config.orientation_tolerance)
    elif config.orientation_tolerance<math.pi:
        raise ValueError("strict orientation acceptance requires observed rotation")
    hold=xp.where(reached,state.hold_count+1,0)
    status=xp.where(state.steps+1>=config.timeout_steps,5,0)
    status=xp.where(hold>=round(.3/config.control_dt),1,status)
    status=xp.where(loss_count>=round(config.contact_loss_seconds/config.control_dt),8,status)
    status=xp.where(peak>config.force_limit,9,status)
    status=xp.where(lateral_peak>config.lateral_force_limit,9,status)
    status=xp.where(forbidden,2,status)
    status=xp.where(~xp.asarray(finite),6,status)
    return status,seen,loss_count,hold,lost
