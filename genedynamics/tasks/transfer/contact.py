"""Backend-neutral contact task and hybrid controller contracts.

Positive command[12] is compressive normal force in N. The normal is world +z;
positive force pushes down. command[0:6] remains world spatial twist. Force
feedback is measured from constraints, never copied from the commanded force.
"""
from dataclasses import dataclass
import math
from typing import Any,NamedTuple
from .runtime import MechanismSpec


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

    def __post_init__(self):
        if self.socket_profile not in {"legacy","matched_keyway"}: raise ValueError("unknown socket profile")
        if not math.isfinite(self.mating_clearance) or not 0<self.mating_clearance<=.002:
            raise ValueError("mating clearance outside declared 0..2 mm pilot support")
        if self.wrench_semantics not in {"legacy_aggregate","tool_at_tcp_world"}: raise ValueError("unknown wrench semantics")
        if self.guide_profile not in {"legacy","spatial_dual_channel"}: raise ValueError("unknown guide profile")
        if self.guide_profile!="legacy" and self.insertion:raise ValueError("spatial guide requires guided contact alignment")
        if self.socket_profile!="legacy" and not self.insertion:
            raise NotImplementedError("matched keyway requires insertion")
        if self.environment not in {"guided_contact_alignment","keyed_insertion_recovery","mobile_keyed_insertion_recovery"}:
            raise ValueError("unknown contact environment")
        if self.robot not in {"panda","xarm7","ur5e","go2_z1"}: raise ValueError("unsupported contact arm")
        if any(not math.isfinite(x) or x<=0 for x in (self.physics_dt,self.control_dt,self.timeout_seconds,
             self.force_limit,self.desired_force,self.force_rate,self.contact_loss_seconds,self.force_integral_limit)):
            raise ValueError("contact limits/clocks must be finite positive")
        if any(not math.isfinite(v) or v<0 for v in (self.force_feedback_gain,self.force_integral_gain)):
            raise ValueError("force feedback gains must be finite nonnegative")
        if not math.isfinite(self.tangential_wrench_limit) or self.tangential_wrench_limit<=0:raise ValueError("invalid tangential wrench limit")
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
    def goal(self): return (.44,-.20,.596-self.insertion_depth) if self.insertion else (.46,-.18,.596 if self.guide_profile=="spatial_dual_channel" else .586)


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
    amplitude=xp.asarray([.08]*3+[.4]*3+[1.]*6+[config.force_limit],dtype=command.dtype)
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


def contact_termination(xp,state,position,normal,peak,forbidden,finite,saturated,config,*,rotation=None,lateral_peak=0.,axial_contact=None,seated_history=None):
    seen=xp.where(state.last_command[12]>.1,state.contact_seen|(normal>config.contact_threshold),False)
    lost=seen&(state.last_command[12]>.1)&(normal<config.contact_threshold)
    loss_count=xp.where(lost,state.loss_count+1,0)
    if config.socket_profile=="matched_keyway" or config.guide_profile=="spatial_dual_channel":
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
