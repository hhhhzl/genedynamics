"""The four existing physical scenes, with honest body and protocol scopes."""
from __future__ import annotations

from dataclasses import replace

from .contracts import TaskSpec

ALIASES = {
    "T1": "obstacle_transfer", "T2": "guided_contact_alignment",
    "T3": "keyed_insertion_recovery", "T4": "mobile_keyed_insertion_recovery",
}


def task_spec(task, *, robot=None, config_overrides=None):
    from genedynamics.learning.cross_embodiment.contracts import ObstacleTransferConfig
    from genedynamics.learning.cross_embodiment.contact import mating_contact_config, spatial_guide_config

    name = ALIASES.get(task, task)
    if name not in ALIASES.values():
        raise ValueError(f"unknown four-task environment: {task}")
    alias = next(key for key, value in ALIASES.items() if name == value)
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
