"""Opt-in T3 extension with a measured, passively translating socket.

The original static task keeps its contract and defaults. This first extension
has one x translation; no unobserved fixture mechanism enters an actor. World
observations and actions remain the public API. The shared controller wrapper
uses the nominal socket chart and converts measured relative velocities back
to world velocities before execution.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import math

import numpy as np

from .contact import ContactConfig, ContactMechanismSpec, mating_contact_config
from .episode import TaskSpec


ENVIRONMENT_NAME = "compliant_keyed_insertion_recovery"
CONTRACT_VERSION = "compliant-keyed-insertion-public-v1"
OBSERVATION_VERSION = "public-fixture-world-v1"
FIXTURE_REST_POSITION = np.asarray((.44, -.20, 0.))
FIXTURE_FIELDS = ("fixture_position", "fixture_rotation", "fixture_twist")


@dataclass(frozen=True)
class CompliantSocketConfig(ContactConfig):
    fixture_profile: str = "passive_x_v1"
    fixture_travel_m: float = .012
    fixture_initial_offset_m: float = 0.
    fixture_meaningful_motion_m: float = .0001
    fixture_stop_margin_m: float = .0005
    fixture_settle_speed_mps: float = .003
    fixture_settle_acceleration_mps2: float = 2.
    fixture_terminal_dwell_seconds: float = .2

    def __post_init__(self):
        super().__post_init__()
        if (self.environment != "keyed_insertion_recovery" or self.mobile
                or self.robot not in {"panda", "xarm7", "ur5e"}
                or self.socket_profile != "matched_keyway"
                or self.fixture_profile != "passive_x_v1"):
            raise ValueError("passive socket requires a fixed-arm matched T3 task")
        if self.reset_joint_noise != 0.:
            raise ValueError("passive socket v1 uses task-space reset noise, not topology-wide q noise")
        if (not math.isfinite(self.fixture_travel_m)
                or not .002 <= self.fixture_travel_m <= .02
                or not math.isfinite(self.fixture_initial_offset_m)
                or abs(self.fixture_initial_offset_m) > self.fixture_travel_m / 2):
            raise ValueError("fixture travel or initial displacement outside development support")
        if (not math.isfinite(self.fixture_meaningful_motion_m)
                or not 0.<self.fixture_meaningful_motion_m<self.fixture_travel_m/2
                or not math.isfinite(self.fixture_stop_margin_m)
                or not 0.<self.fixture_stop_margin_m<self.fixture_travel_m/2
                or not math.isfinite(self.fixture_settle_speed_mps)
                or self.fixture_settle_speed_mps<=0.
                or not math.isfinite(self.fixture_settle_acceleration_mps2)
                or self.fixture_settle_acceleration_mps2<=0.
                or not math.isfinite(self.fixture_terminal_dwell_seconds)
                or self.fixture_terminal_dwell_seconds<self.control_dt):
            raise ValueError("closed-loop fixture evidence thresholds are invalid")


@dataclass(frozen=True)
class CompliantSocketMechanismSpec(ContactMechanismSpec):
    """Construction-only passive mechanism; excluded from public observations."""
    fixture_mass_kg: float = .5
    fixture_stiffness_Npm: float = 800.
    fixture_damping_Nspm: float = 12.

    def __post_init__(self):
        super().__post_init__()
        if any(not math.isfinite(value) or value <= 0 for value in (
                self.fixture_mass_kg, self.fixture_stiffness_Npm,
                self.fixture_damping_Nspm)):
            raise ValueError("passive fixture mass, spring and damping must be positive")


@dataclass(frozen=True)
class CompliantSocketTaskSpec(TaskSpec):
    def describe(self):
        value = super().describe()
        value["contract_version"] = CONTRACT_VERSION
        value["observation_version"] = OBSERVATION_VERSION
        value["observations"].update({
            "joint_position": "robot encoders only; passive fixture coordinates excluded",
            "joint_velocity": "robot encoder velocities only; fixture velocity excluded",
            "goal_position": "current measured fixture-relative seat in world coordinates, m",
            "fixture_position": "measured socket task-frame origin in world xyz, m",
            "fixture_rotation": "measured world-from-socket rotation; identity for passive_x_v1",
            "fixture_twist": "measured world socket linear/angular velocity, m/s then rad/s",
        })
        value["excluded_policy_fields"] += [
            "fixture_mass_kg", "fixture_stiffness_Npm", "fixture_damping_Nspm",
            "fixture_passive_force", "fixture_joint_coordinate",
        ]
        value["fixture_rest_position_m"] = FIXTURE_REST_POSITION.tolist()
        value["scope"] = "single-axis compliant fixture development task; no old static result reuse"
        return value


def compliant_socket_spec(*, robot="xarm7", config_overrides=None):
    values = asdict(mating_contact_config(robot))
    overrides = dict(config_overrides or {})
    forbidden = set(overrides) & {
        "environment", "robot", "mobile", "socket_profile", "wrench_semantics",
        "guide_profile", "fixture_profile",
    }
    if forbidden:
        raise ValueError(f"dynamic task identity cannot be overridden: {sorted(forbidden)}")
    values.update(overrides)
    return CompliantSocketTaskSpec(
        name=ENVIRONMENT_NAME, alias="T3", config=CompliantSocketConfig(**values),
        family="contact", body_scope="fixed arm with one passive spring-damper socket translation",
        mode_semantics=("straight withdrawal relative to socket", "lateral withdrawal relative to socket"),
    )


def validate_fixture_observation(public):
    """Validate the same measured fixture chart in execution and offline audit."""
    if any(name not in public for name in FIXTURE_FIELDS):
        raise ValueError("measured fixture pose and twist are required")
    position = np.asarray(public["fixture_position"])
    rotation = np.asarray(public["fixture_rotation"])
    twist = np.asarray(public["fixture_twist"])
    if (position.ndim != 2 or position.shape[1] != 3 or len(position) == 0
            or rotation.shape != (len(position), 3, 3)
            or twist.shape != (len(position), 6)
            or not np.isfinite(np.r_[position.ravel(), rotation.ravel(), twist.ravel()]).all()
            or not np.allclose(rotation, np.eye(3), atol=1e-7)
            or np.max(np.abs(position[:, 1:] - FIXTURE_REST_POSITION[1:])) > 1e-7
            or np.max(np.abs(twist[:, 1:])) > 1e-7):
        raise ValueError("passive_x_v1 requires finite identity-rotation x translation")
    return position, rotation, twist


def fixture_relative_observation(public):
    """Map measured world state into the nominal socket chart, without truth.

    This is a shared execution adapter for the one-axis extension, not a claim
    that an old learned checkpoint has the new response-model semantics.
    Previous requests/issued actions remain recorded in world coordinates: they
    are historical commands and cannot be converted using today's velocity.
    """
    position, _, twist = validate_fixture_observation(public)
    result = {key: np.asarray(value).copy() for key, value in public.items()
              if key not in FIXTURE_FIELDS}
    displacement = position - FIXTURE_REST_POSITION
    for key in ("tcp_position", "reference_position", "goal_position"):
        result[key] = result[key] - displacement
    result["tcp_twist"] = result["tcp_twist"] - twist
    return result


class FixtureRelativePolicy:
    """Shared task-frame wrapper, including measured translational feedforward."""
    VERSION = "passive-socket-relative-policy-v1"

    def __init__(self, actor):
        self.actor = actor

    def act(self, public):
        command = np.asarray(self.actor.act(fixture_relative_observation(public))).copy()
        command[..., :3] += np.asarray(public["fixture_twist"])[..., :3]
        return command

    def reset(self, **kwargs):
        return self.actor.reset(**kwargs)

    def get_state(self):
        return dict(version=self.VERSION, actor=deepcopy(self.actor.get_state()))

    def restore(self, state):
        if state.get("version") != self.VERSION:
            raise ValueError("fixture wrapper continuation version differs")
        self.actor.restore(deepcopy(state["actor"]))

    def reorder(self, indices):
        return self.actor.reorder(indices)


def evaluate_compliant_socket_records(records, spec):
    """Evaluate relative insertion plus passive-motion and joint settling evidence."""
    from .evaluation import aggregate_episode_rows, evaluate_records

    converted = deepcopy(records)
    excursions, speeds, exact_rows = [], [], []
    for original, record in zip(records, converted):
        offsets, velocities = [], []
        for row in record.traces:
            if not row["valid"]:
                continue
            if any(name not in row for name in FIXTURE_FIELDS):
                raise ValueError("dynamic evaluator requires recorded fixture observations")
            position, _, twist = validate_fixture_observation({
                name: np.asarray(row[name])[None] for name in FIXTURE_FIELDS
            })
            offset = position[0] - FIXTURE_REST_POSITION
            row["tcp_position"] = np.asarray(row["tcp_position"]) - offset
            offsets.append(abs(float(offset[0])))
            velocities.append(abs(float(twist[0, 0])))
        excursions.append(max(offsets, default=0.))
        speeds.append(max(velocities, default=0.))
    result = evaluate_records(converted, replace(spec, name="keyed_insertion_recovery"))
    for original, row, excursion, speed in zip(records,result["episodes"],excursions,speeds):
        traces=[trace for trace in original.traces if trace["valid"]]
        required=("fixture_position_substeps","fixture_twist_substeps","fixture_stop_substeps",
                  "tcp_position_substeps","normal_force_substeps","axial_support_substeps",
                  "force_peak_substeps","lateral_force_substeps","contact_region_substeps")
        exact=bool(traces) and all(all(name in trace for name in required) for trace in traces)
        closed_loop=False;motion_index=contact_index=recovery_index=insertion_index=None
        terminal_stable=False;stop_contact=False;peak_acceleration=None
        if exact:
            widths={len(np.asarray(trace["fixture_stop_substeps"])) for trace in traces}
            if widths!={spec.config.substeps}:
                raise ValueError("compliant fixture evidence has the wrong substep count")
            fixture=np.concatenate([np.asarray(trace["fixture_position_substeps"],float).reshape(-1,3) for trace in traces])
            twist=np.concatenate([np.asarray(trace["fixture_twist_substeps"],float).reshape(-1,6) for trace in traces])
            stops=np.concatenate([np.asarray(trace["fixture_stop_substeps"],bool) for trace in traces])
            tcp=np.concatenate([np.asarray(trace["tcp_position_substeps"],float).reshape(-1,3) for trace in traces])
            normal=np.concatenate([np.asarray(trace["normal_force_substeps"],float) for trace in traces])
            axial=np.concatenate([np.asarray(trace["axial_support_substeps"],bool) for trace in traces])
            force=np.concatenate([np.asarray(trace["force_peak_substeps"],float) for trace in traces])
            lateral=np.concatenate([np.asarray(trace["lateral_force_substeps"],float) for trace in traces])
            regions=np.concatenate([np.asarray(trace["contact_region_substeps"],np.uint8) for trace in traces])
            if (not np.isfinite(np.r_[fixture.ravel(),twist.ravel(),tcp.ravel(),normal,force,lateral]).all()
                    or fixture.shape!=(len(stops),3) or twist.shape!=(len(stops),6)
                    or tcp.shape!=(len(stops),3)):
                raise ValueError("compliant fixture substep evidence is nonfinite or shape-inconsistent")
            offset=fixture[:,0]-FIXTURE_REST_POSITION[0]
            acceleration=np.diff(twist[:,0],prepend=twist[:1,0])/spec.config.physics_dt
            peak_acceleration=float(np.max(np.abs(acceleration)))
            stop_contact=bool(np.any(stops)|np.any(np.abs(offset)>=spec.config.fixture_travel_m-spec.config.fixture_stop_margin_m))
            motion=np.abs(offset-offset[0])>=spec.config.fixture_meaningful_motion_m
            ids=np.flatnonzero(motion);motion_index=int(ids[0]) if len(ids) else None
            contact=axial&(normal>spec.config.contact_threshold)&((regions&8)!=0)
            ids=np.flatnonzero(contact);contact_index=int(ids[0]) if len(ids) else None
            relative=tcp-(fixture-FIXTURE_REST_POSITION)
            if contact_index is not None:
                retreat=(relative[:,2]-relative[contact_index,2])>=.001
                unload=normal<=max(spec.config.contact_threshold,float(normal[contact_index])-1.)
                recovery_lower=max(contact_index,-1 if motion_index is None else motion_index)
                ids=np.flatnonzero((retreat|unload)&(np.arange(len(normal))>recovery_lower))
                recovery_index=int(ids[0]) if len(ids) else None
            goal=np.asarray(spec.config.goal)
            inserted=(np.linalg.norm(relative-goal,axis=1)<spec.config.position_tolerance)
            lower=-1 if recovery_index is None else recovery_index
            ids=np.flatnonzero(inserted&(np.arange(len(inserted))>lower))
            insertion_index=int(ids[0]) if len(ids) else None
            dwell=max(1,round(spec.config.fixture_terminal_dwell_seconds/spec.config.physics_dt))
            start=max(0,len(stops)-dwell)
            terminal_stable=(len(stops)>=dwell and insertion_index is not None
                and np.all(inserted[start:]) and np.all(np.abs(twist[start:,0])<=spec.config.fixture_settle_speed_mps)
                and np.all(np.abs(acceleration[start:])<=spec.config.fixture_settle_acceleration_mps2)
                and np.all(np.abs(normal[start:]-spec.config.desired_force)<2.5)
                and np.all(force[start:]<=spec.config.force_limit)
                and np.all(lateral[start:]<=spec.config.lateral_force_limit)
                and not np.any(stops[start:]))
            ordered=(motion_index is not None and contact_index is not None
                     and contact_index<motion_index
                     and recovery_index is not None and insertion_index is not None
                     and motion_index<=recovery_index<insertion_index)
            closed_loop=bool(row["success"] and ordered
                             and recovery_index is not None and insertion_index is not None
                             and not stop_contact and terminal_stable)
        if not exact:outcome="missing_physics_substep_evidence"
        elif stop_contact:outcome="fixture_travel_stop"
        elif motion_index is None:outcome="fixture_motion_not_excited"
        elif contact_index is None:outcome="loaded_contact_not_established"
        elif recovery_index is None:outcome="relative_recovery_missing"
        elif insertion_index is None:outcome="relative_insertion_missing"
        elif not terminal_stable:outcome="fixture_connector_not_stable"
        elif not row["success"]:outcome=row["outcome"]
        else:outcome="success"
        row["evaluation_version"] = CONTRACT_VERSION
        row["fixture_metric_sampling"] = "physics substeps when present; endpoint metrics retained for compatibility"
        row["fixture_max_displacement_m"] = excursion
        row["fixture_peak_speed_mps"] = speed
        row.update(success=closed_loop,constrained_success=closed_loop,
                   closed_loop_physical_success=closed_loop,
                   task_cost=0. if closed_loop else 1.,
                   capped_normalized_completion_time=row["capped_normalized_completion_time"] if closed_loop else 1.,
                   outcome=outcome,constraint_violation=bool(row["constraint_violation"] or (exact and stop_contact)),
                   exact_physics_fixture_evidence=exact,fixture_motion_physics_step=motion_index,
                   loaded_contact_physics_step=contact_index,recovery_physics_step=recovery_index,
                   relative_insertion_physics_step=insertion_index,fixture_stop_contact=stop_contact,
                   terminal_joint_stability_passed=bool(terminal_stable),fixture_peak_acceleration_mps2=peak_acceleration)
        exact_rows.append(row)
    report=aggregate_episode_rows(exact_rows,scope="independent moving-fixture-frame closed-loop evaluator; every requested episode retained")
    report["evaluation_version"]=CONTRACT_VERSION
    report["observation_version"]=OBSERVATION_VERSION
    report["fixture_metric_sampling"]="physics substeps required for closed-loop success"
    return report
