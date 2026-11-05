"""The four existing physical scenes, with honest body and protocol scopes."""
from __future__ import annotations

from dataclasses import replace
import math

from .episode import TaskSpec
from .task_identity import (
    LEGACY_TO_CANONICAL,
    canonical_task_id,
    legacy_task_id,
)

# Compatibility export for historical callers.  New dispatch uses semantic
# names and converts aliases only at reader boundaries.
ALIASES = dict(LEGACY_TO_CANONICAL)


SHARED_EXECUTION_VERSIONS = {
    "T1": "four-task-shared-execution-v3",
    "T2": "four-task-shared-execution-v7",
    "T3": "four-task-shared-execution-v3",
    "T4": "four-task-shared-execution-v3",
}


def validate_shared_execution(spec, mechanism, values=None):
    """Validate the method-independent execution policy inherited by every row.

    This stays beside the task factory because it is part of the public task
    binding, not a solver or baseline option.  Exact per-task dictionaries make
    accidental method-specific overrides fail before any physics is executed.
    """
    values = dict(values or {})
    if not values:
        return {}
    alias = spec.alias
    expected_version = SHARED_EXECUTION_VERSIONS[alias]
    if values.pop("version", None) != expected_version:
        raise ValueError(
            f"{alias} shared_execution.version must be {expected_version!r}"
        )
    expected_keys = {
        "T1": set(),
        "T2": {
            "strict_evaluator_version", "physics_dt_seconds",
            "control_dt_seconds", "contact_time_constant_seconds",
            "contact_damping_ratio", "execution_profile",
            "guide_mode0_transit_speed_mps",
            "guide_mode0_post_outlet_route",
            "guide_reference_admission",
            "guide_controller_version",
            "guide_route_version",
            "guide_clearance_route_version",
            "terminal_actor_interface",
            "terminal_actor_interface_parameters",
            "surface_force_target_N",
            "terminal_surface_force_target_N",
            "mode0_preoutlet_stiffness_ramp_m",
            "mode0_transition_spring_budget_N",
            "log_stiffness_limit",
            "tangential_wrench_limit_N",
            "timeout_seconds",
            "terminal_position_tolerance_m",
            "terminal_orientation_tolerance_rad",
            "terminal_dwell_controls",
            "force_limit_N",
            "lateral_force_limit_N",
        },
        "T3": {"insertion_lead_z_m", "retreat_release_version"},
        "T4": {
            "execution_profile", "reset_profile", "reset_joint_noise_rad",
            "body_intent_profile", "body_velocity_world",
            "body_intent_start_seconds", "body_intent_duration_seconds",
        },
    }[alias]
    if set(values) != expected_keys:
        raise ValueError(
            f"{alias} shared_execution fields differ: expected {sorted(expected_keys)}"
        )
    if alias == "T1":
        return {"version": expected_version}
    if alias == "T2":
        if values["strict_evaluator_version"] != "strict-guide-substep-v4":
            raise ValueError("T2 requires the frozen physics-substep evaluator")
        required = {
            "physics_dt_seconds": spec.config.physics_dt,
            "control_dt_seconds": spec.config.control_dt,
            "contact_time_constant_seconds": mechanism.contact_time_constant,
            "contact_damping_ratio": 2.0,
            "execution_profile": spec.config.execution_profile,
            "guide_mode0_transit_speed_mps": (
                spec.config.guide_mode0_transit_speed_mps
            ),
            "guide_mode0_post_outlet_route": (
                spec.config.guide_mode0_post_outlet_route
            ),
            "guide_reference_admission": spec.config.guide_reference_admission,
            "guide_controller_version": (
                "public-T2-response-normalized-hybrid-force-v50"
            ),
            "guide_route_version": "public-T2-response-normalized-guide-route-v4",
            "guide_clearance_route_version": (
                "public-T2-response-normalized-clearance-traverse-route-v3"
            ),
            "terminal_actor_interface": (
                "public-T2-completion-budget-causal-traverse-v18"
            ),
            "terminal_actor_interface_parameters": {
                "completion_reserve_controls": 150,
                "reserve_taper_controls": 100,
                "reference_lag_full_authority_m": .008,
                "reference_lag_zero_authority_m": .012,
                "maximum_authority_decay_per_control": .05,
                "completion_progress_floor_mps": .02,
                "post_traverse_reserve_controls": 100,
                "completion_budget_taper_controls": 60,
                "force_speed_full_authority_mps": .01,
                "force_speed_zero_authority_mps": .025,
            },
            "surface_force_target_N": spec.config.desired_force,
            "terminal_surface_force_target_N": 6.0,
            "mode0_preoutlet_stiffness_ramp_m": (
                spec.config.position_tolerance
            ),
            "mode0_transition_spring_budget_N": 8.8,
            "log_stiffness_limit": spec.config.log_stiffness_limit,
            "tangential_wrench_limit_N": spec.config.tangential_wrench_limit,
            "timeout_seconds": spec.config.timeout_seconds,
            "terminal_position_tolerance_m": spec.config.position_tolerance,
            "terminal_orientation_tolerance_rad": spec.config.orientation_tolerance,
            "terminal_dwell_controls": 15,
            "force_limit_N": spec.config.force_limit,
            "lateral_force_limit_N": spec.config.lateral_force_limit,
        }
        if any(values[key] != value for key, value in required.items()):
            raise ValueError("T2 shared execution disagrees with constructed physics")
        if required != {
            "physics_dt_seconds": .001,
            "control_dt_seconds": .02,
            "contact_time_constant_seconds": .04,
            "contact_damping_ratio": 2.0,
            "execution_profile": "substep_wrench_governor",
            "guide_mode0_transit_speed_mps": .00675,
            "guide_mode0_post_outlet_route": "concurrent_monotone_lift_v3",
            "guide_reference_admission": "response_normalized_completion",
            "guide_controller_version": (
                "public-T2-response-normalized-hybrid-force-v50"
            ),
            "guide_route_version": "public-T2-response-normalized-guide-route-v4",
            "guide_clearance_route_version": (
                "public-T2-response-normalized-clearance-traverse-route-v3"
            ),
            "terminal_actor_interface": (
                "public-T2-completion-budget-causal-traverse-v18"
            ),
            "terminal_actor_interface_parameters": {
                "completion_reserve_controls": 150,
                "reserve_taper_controls": 100,
                "reference_lag_full_authority_m": .008,
                "reference_lag_zero_authority_m": .012,
                "maximum_authority_decay_per_control": .05,
                "completion_progress_floor_mps": .02,
                "post_traverse_reserve_controls": 100,
                "completion_budget_taper_controls": 60,
                "force_speed_full_authority_mps": .01,
                "force_speed_zero_authority_mps": .025,
            },
            "surface_force_target_N": 5.0,
            "terminal_surface_force_target_N": 6.0,
            "mode0_preoutlet_stiffness_ramp_m": .002,
            "mode0_transition_spring_budget_N": 8.8,
            "log_stiffness_limit": 1.5,
            "tangential_wrench_limit_N": 16.,
            "timeout_seconds": 20.,
            "terminal_position_tolerance_m": .002,
            "terminal_orientation_tolerance_rad": math.radians(2.),
            "terminal_dwell_controls": 15,
            "force_limit_N": 25.,
            "lateral_force_limit_N": 10.,
        }:
            raise ValueError("T2 causal-traverse shared execution v7 is not active")
    elif alias == "T3":
        if (not math.isclose(float(values["insertion_lead_z_m"]), .015)
                or values["retreat_release_version"] != "safe_settled_v1"):
            raise ValueError("T3 requires the frozen 15 mm lead and safe-settled release")
    else:
        velocity = tuple(float(value) for value in values["body_velocity_world"])
        if (
            values["execution_profile"] != spec.config.execution_profile
            or values["reset_profile"] != spec.config.reset_profile
            or float(values["reset_joint_noise_rad"]) != spec.config.reset_joint_noise
            or values["body_intent_profile"] != "public_planar_v1"
            or velocity != (.005, .005, 0.)
            or float(values["body_intent_start_seconds"]) != 0.
            or float(values["body_intent_duration_seconds"]) != 6.
        ):
            raise ValueError("T4 shared reset/body intent disagrees with the frozen contract")
        if spec.config.execution_profile != "joint_margin_public_reference":
            raise ValueError("T4 requires the public joint-margin execution profile")
    return {"version": expected_version, **values}


def mechanism_spec(task, values=None, *, body_support_parameters=None):
    """Construct the task's hidden mechanism without exposing it to actors."""
    from genedynamics.tasks.transfer.runtime import MechanismSpec
    from .compliant_socket import ENVIRONMENT_NAME, CompliantSocketMechanismSpec
    if task in {ENVIRONMENT_NAME,"compliant_fixture_insertion"}:
        if body_support_parameters:
            raise ValueError("fixed-arm compliant socket has no body-support parameters")
        return CompliantSocketMechanismSpec(**dict(values or {}))
    from genedynamics.tasks.transfer.contact import (
        ContactMechanismSpec,
        MobileContactMechanismSpec,
    )

    if task=="aperture_brake":
        if body_support_parameters:
            raise ValueError("fixed-arm aperture transfer has no body-support parameters")
        return MechanismSpec(**dict(values or {}))
    if task=="corner_guided_seating":
        if body_support_parameters:
            raise ValueError("fixed-arm corner guide has no body-support parameters")
        return ContactMechanismSpec(**dict(values or {}))

    alias = legacy_task_id(task)
    parameters = dict(values or {})
    support = dict(body_support_parameters or {})
    if alias == "T4":
        if "body_servo_gain" in support:
            parameters["body_servo_gain"] = support["body_servo_gain"]
        return MobileContactMechanismSpec(**parameters)
    if alias in {"T2", "T3"}:
        return ContactMechanismSpec(**parameters)
    return MechanismSpec(**parameters)


def task_spec(task, *, robot=None, config_overrides=None):
    from genedynamics.tasks.transfer.runtime import ObstacleTransferConfig
    from genedynamics.tasks.transfer.contact import mating_contact_config, spatial_guide_config, corner_guide_config
    from .compliant_socket import ENVIRONMENT_NAME, compliant_socket_spec
    if task in {ENVIRONMENT_NAME,"compliant_fixture_insertion"}:
        spec=compliant_socket_spec(robot=robot or "xarm7", config_overrides=config_overrides)
        if task=="compliant_fixture_insertion":
            spec=replace(spec,name=task,qualification="semantic CPU closed-loop qualification")
        return spec
    if task == "aperture_brake":
        from genedynamics.tasks.transfer.runtime import ObstacleTransferConfig
        if config_overrides:
            forbidden = set(config_overrides) & {"robot", "scene_profile"}
            if forbidden:
                raise ValueError(f"task identity/profile cannot be overridden: {sorted(forbidden)}")
        config = ObstacleTransferConfig(
            robot=robot or "xarm7", start_position=(.38, -.09, .55),
            goal_position=(.64, -.09, .55), goal_yaw=0.,
            scene_profile="aperture_brake_v1", timeout_seconds=12.,
        )
        if config_overrides:
            config = replace(config, **dict(config_overrides))
        return TaskSpec(
            name=task, alias="T1", config=config, family="obstacle",
            body_scope="fixed arm carrying a rigid L-shaped object through a solid framed aperture",
            mode_semantics=("lower-side approach", "upper-side approach"),
            qualification="semantic CPU whole-envelope aperture qualification",
        )
    if task=="corner_guided_seating":
        if config_overrides:
            forbidden=set(config_overrides)&{"robot","environment","mobile","socket_profile","guide_profile","wrench_semantics"}
            if forbidden:
                raise ValueError(f"task identity/profile cannot be overridden: {sorted(forbidden)}")
        config=corner_guide_config(robot or "xarm7")
        if config_overrides:
            config=replace(config,**dict(config_overrides))
        return TaskSpec(name=task,alias="T2",config=config,family="contact",
                        body_scope="fixed arm with solid planar right-angle guide and end stop",
                        mode_semantics=("right-angle corner traversal",))

    name = canonical_task_id(task)
    alias = legacy_task_id(name)
    robot = robot or ("go2_z1" if alias == "T4" else "xarm7")
    if alias == "T1":
        config = ObstacleTransferConfig(robot=robot)
        family, body_scope = "obstacle", "fixed arm with conservative collision proxies"
        modes = ("upper route", "side route")
    elif alias == "T2":
        config = spatial_guide_config(robot)
        family, body_scope = "contact", "fixed arm, spatial dual-channel guide"
        modes = ("x-first channel", "y-first channel")
    else:
        # Go2 has no qualified noisy task-space reset. Keep this explicit
        # rather than claiming it covers the fixed-arm reset distribution.
        config = mating_contact_config("xarm7" if robot == "go2_z1" else robot)
        if alias == "T4":
            config = replace(config, environment=name, robot=robot, mobile=True,
                             reset_task_noise=0. if robot == "go2_z1" else config.reset_task_noise)
        family = "contact"
        body_scope = ("floating Go2+Z1 with host OSQP stance WBC; body qualification pending"
                      if robot == "go2_z1" else "planar mobile proxy; not a floating/wheeled robot qualification"
                      if alias == "T4" else "fixed arm, matched keyway and TCP wrench")
        modes = ("straight withdrawal recovery", "lateral withdrawal recovery")
    if config_overrides:
        forbidden = set(config_overrides) & {"robot", "environment", "mobile", "socket_profile", "guide_profile", "wrench_semantics"}
        if forbidden:
            raise ValueError(f"task identity/profile cannot be overridden: {sorted(forbidden)}")
        config = replace(config, **dict(config_overrides))
    return TaskSpec(name=name, alias=alias, config=config, family=family,
                    body_scope=body_scope, mode_semantics=modes)
