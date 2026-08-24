"""Simulation-facing robot descriptions shared by tasks and controllers.

Profiles are construction-time metadata.  Names are resolved once before a
Brax/MJX step is jitted; runtime code consumes :class:`RobotBinding` indices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, FrozenSet, Iterable, Mapping, Optional, Tuple


FIXED_BASE = "fixed_base"
FLOATING_BASE = "floating_base"
CARTESIAN_JACOBIAN = "cartesian_jacobian"
TORQUE_CONTROL = "torque_control"
SINGLE_TOOL = "single_tool"
BIPED = "biped"
TWO_HANDS = "two_hands"
TWO_FEET = "two_feet"
WHOLE_BODY_CONTROL = "whole_body_control"


@dataclass(frozen=True)
class NamedElement:
    """A semantic robot role bound to one named MuJoCo element."""

    kind: str  # ``body`` | ``site`` | ``geom``
    name: str


@dataclass(frozen=True)
class RobotProfile:
    """Robot-only topology and semantics required by task composition."""

    model_id: str
    robot_type: str
    base_type: str
    model_path_resolver: Callable[[], str]
    actuated_joints: Tuple[str, ...]
    actuator_names: Tuple[str, ...] = ()
    joint_groups: Mapping[str, Tuple[int, ...]] = field(default_factory=dict)
    elements: Mapping[str, NamedElement] = field(default_factory=dict)
    capabilities: FrozenSet[str] = frozenset()
    scene_resolvers: Mapping[str, Callable[[], str]] = field(default_factory=dict)
    home_keyframe: str = "home"
    controller_defaults: Mapping[str, object] = field(default_factory=dict)

    @property
    def num_actuated(self) -> int:
        return len(self.actuated_joints)

    def model_path(self, task: Optional[str] = None) -> str:
        if task is not None and task in self.scene_resolvers:
            return str(self.scene_resolvers[task]())
        return str(self.model_path_resolver())

    def require(self, required: Iterable[str]) -> None:
        missing = frozenset(required) - self.capabilities
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(
                f"robot '{self.model_id}' is incompatible; missing capabilities: {names}"
            )


@dataclass(frozen=True)
class RobotBinding:
    """Resolved model indices for one robot in a compiled task scene."""

    profile: RobotProfile
    qpos_indices: Tuple[int, ...]
    dof_indices: Tuple[int, ...]
    actuator_indices: Tuple[int, ...]
    element_ids: Mapping[str, int]

    @classmethod
    def from_mujoco_model(cls, profile: RobotProfile, model: object) -> "RobotBinding":
        """Resolve and validate a profile against a ``mujoco.MjModel``."""
        import mujoco

        qpos = []
        dofs = []
        for name in profile.actuated_joints:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise KeyError(f"{profile.model_id} joint not found in composed MJCF: {name}")
            qpos.append(int(model.jnt_qposadr[jid]))
            dofs.append(int(model.jnt_dofadr[jid]))

        actuator_names = profile.actuator_names or profile.actuated_joints
        actuators = []
        for name in actuator_names:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise KeyError(f"{profile.model_id} actuator not found in composed MJCF: {name}")
            actuators.append(int(aid))

        obj_types = {
            "body": mujoco.mjtObj.mjOBJ_BODY,
            "site": mujoco.mjtObj.mjOBJ_SITE,
            "geom": mujoco.mjtObj.mjOBJ_GEOM,
        }
        element_ids = {}
        for role, element in profile.elements.items():
            if element.kind not in obj_types:
                raise ValueError(f"unsupported MuJoCo element kind: {element.kind}")
            idx = mujoco.mj_name2id(model, obj_types[element.kind], element.name)
            if idx < 0:
                raise KeyError(
                    f"{profile.model_id} role '{role}' missing {element.kind}: {element.name}"
                )
            element_ids[role] = int(idx)

        if len(actuators) != profile.num_actuated:
            raise ValueError(
                f"{profile.model_id} has {len(actuators)} bound actuators for "
                f"{profile.num_actuated} actuated joints"
            )
        return cls(
            profile=profile,
            qpos_indices=tuple(qpos),
            dof_indices=tuple(dofs),
            actuator_indices=tuple(actuators),
            element_ids=element_ids,
        )

    def group(self, name: str) -> Tuple[int, ...]:
        try:
            return tuple(self.profile.joint_groups[name])
        except KeyError as exc:
            raise KeyError(f"{self.profile.model_id} has no joint group '{name}'") from exc
