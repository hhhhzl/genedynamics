"""Component registries for the deploy pipeline.

Each major deploy concern (IO, controller, follower, safety filter, task,
observer) gets its own :class:`BaseRegistry` instance keyed by string. The
nested-class config layer in :mod:`genedynamics.deploy.config_schema`
resolves these keys via :func:`initialize_class`, so adding a new
controller is one ``register`` call here plus one config-line swap on the
preset side.

Why a separate module from ``core/registry/``? The core registries
(``backends``, ``solvers``, ``environments``) are intentionally low-level
and use a two-level ``{type: {backend: impl}}`` structure. Deploy
components only need a flat ``{name: cls}`` map and can reuse
:class:`BaseRegistry` directly. Keeping them in their own module isolates
the dependency surface — none of these registries need to be loaded by
``core``.

Default registrations are performed at import time so simply doing
``from genedynamics.deploy import registries`` is enough to get the canonical
keys (``io.mujoco``, ``controller.wbc``, etc.) populated. To add a custom
component without forking the package, call e.g.
``registries.controller_registry.register("my_ctl", MyController)``.
"""

from __future__ import annotations

from typing import Any, Optional

from genedynamics.registry_base import BaseRegistry

__all__ = [
    "io_registry",
    "controller_registry",
    "follower_registry",
    "safety_registry",
    "task_registry",
    "observer_registry",
    "loco_client_registry",
    "register_defaults",
    "get_registry",
]


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

io_registry: BaseRegistry = BaseRegistry("genedynamics.deploy.io")
controller_registry: BaseRegistry = BaseRegistry("genedynamics.deploy.controllers")
follower_registry: BaseRegistry = BaseRegistry("genedynamics.deploy.followers")
safety_registry: BaseRegistry = BaseRegistry("genedynamics.deploy.safety")
task_registry: BaseRegistry = BaseRegistry("genedynamics.deploy.tasks")
observer_registry: BaseRegistry = BaseRegistry("genedynamics.deploy.observers")
loco_client_registry: BaseRegistry = BaseRegistry("genedynamics.deploy.loco_clients")


_KIND_TO_REGISTRY: dict[str, BaseRegistry] = {
    "io": io_registry,
    "controller": controller_registry,
    "follower": follower_registry,
    "safety": safety_registry,
    "task": task_registry,
    "observer": observer_registry,
    "loco_client": loco_client_registry,
}


def get_registry(key: str) -> BaseRegistry:
    """Look up a registry by ``"<kind>.<name>"`` key.

    Example::

        get_registry("controller.wbc") is controller_registry  # True
    """
    kind = key.split(".", 1)[0]
    reg = _KIND_TO_REGISTRY.get(kind)
    if reg is None:
        raise ValueError(
            f"Unknown registry kind in key {key!r}. "
            f"Expected one of {sorted(_KIND_TO_REGISTRY)}."
        )
    return reg


def _resolve(key: str) -> Optional[type]:
    """Resolve ``"<kind>.<name>"`` to the registered implementation class."""
    kind, name = key.split(".", 1)
    reg = _KIND_TO_REGISTRY.get(kind)
    if reg is None:
        return None
    return reg.get_class(name)


def resolve_class(key: str) -> type:
    """Like :func:`_resolve` but raises ``KeyError`` when the key is unknown."""
    cls = _resolve(key)
    if cls is None:
        # A default can be skipped during the package's first import when an
        # optional component participates in a circular import. At actual
        # resolution time the module graph is complete, so retry once.
        register_defaults(force=True)
        cls = _resolve(key)
    if cls is None:
        kind = key.split(".", 1)[0]
        reg = _KIND_TO_REGISTRY.get(kind)
        avail = reg.list_available() if reg else []
        raise KeyError(f"Registry key not found: {key!r}. Available {kind}.*: {avail}")
    return cls


# ---------------------------------------------------------------------------
# Default registrations
# ---------------------------------------------------------------------------

_DEFAULTS_REGISTERED = False


def register_defaults(*, force: bool = False) -> None:
    """Populate every registry with the canonical built-in implementations.

    Idempotent — safe to call multiple times. Each implementation is loaded
    inside its own ``try`` block so optional heavy deps (mjx, jax, mujoco,
    osqp) do not break import on dev laptops.
    """
    global _DEFAULTS_REGISTERED
    if _DEFAULTS_REGISTERED and not force:
        return

    # ----- io -------------------------------------------------------------
    try:
        from genedynamics.deploy.io.mujoco_io import MujocoRobotIO

        io_registry.register("mujoco", MujocoRobotIO)
    except ImportError:
        pass
    try:
        from genedynamics.deploy.io.mjx_io import MjxRobotIO

        io_registry.register("mjx", MjxRobotIO)
    except ImportError:
        pass
    try:
        from genedynamics.deploy.io.unitree_g1_io import UnitreeG1RobotIO

        io_registry.register("unitree_g1", UnitreeG1RobotIO)
    except ImportError:
        pass
    try:
        from genedynamics.deploy.io.stub_io import StubRobotIO

        io_registry.register("stub", StubRobotIO)
    except ImportError:
        pass
    try:
        from genedynamics.deploy.io.brax_io import BraxRobotIO

        io_registry.register("brax", BraxRobotIO)
    except ImportError:
        pass
    try:
        from genedynamics.deploy.io.isaac_lab_io import IsaacLabRobotIO

        io_registry.register("isaac_lab", IsaacLabRobotIO)
    except ImportError:
        pass

    # ----- controllers ----------------------------------------------------
    try:
        from genedynamics.deploy.controllers.wbc.controller import (
            HumanoidWBCController,
        )

        controller_registry.register("wbc", HumanoidWBCController)
    except ImportError:
        pass
    try:
        from genedynamics.deploy.controllers.sport_mode.controller import (
            SportModeController,
        )

        controller_registry.register("sport_mode", SportModeController)
    except ImportError:
        pass
    try:
        from genedynamics.deploy.controllers.rl.unitree_rl_gym import (
            UnitreeRLGymG1Controller,
        )

        controller_registry.register("rl_unitree_rl_gym", UnitreeRLGymG1Controller)
    except ImportError:
        pass
    try:
        from genedynamics.deploy.controllers.rl.passthrough import (
            PassthroughRLController,
        )

        controller_registry.register("rl_passthrough", PassthroughRLController)
    except ImportError:
        pass
    try:
        from genedynamics.deploy.controllers.quadruped_stepping.controller import (
            QuadrupedSteppingController,
        )

        controller_registry.register("quadruped_stepping", QuadrupedSteppingController)
    except ImportError:
        pass

    # ----- loco clients (sport mode) --------------------------------------
    try:
        from genedynamics.deploy.controllers.sport_mode.real_loco_client import (
            RealLocoClient,
        )

        loco_client_registry.register("real", RealLocoClient)
    except ImportError:
        pass
    try:
        from genedynamics.deploy.controllers.sport_mode.spark_rl_loco_client import (
            SparkRLLocoClient,
        )

        loco_client_registry.register("spark_rl", SparkRLLocoClient)
    except ImportError:
        pass

    # ----- safety ---------------------------------------------------------
    try:
        from genedynamics.deploy.safety.joint_limit import JointLimitFilter
        from genedynamics.deploy.safety.torque_limit import TorqueLimitFilter
        from genedynamics.deploy.safety.self_collision import SelfCollisionFilter
        from genedynamics.deploy.safety.composite import CompositeSafetyFilter

        safety_registry.register("joint_limit", JointLimitFilter)
        safety_registry.register("torque_limit", TorqueLimitFilter)
        safety_registry.register("self_collision", SelfCollisionFilter)
        safety_registry.register("composite", CompositeSafetyFilter)
    except ImportError:
        pass
    try:
        from genedynamics.deploy.safety.cbf_filter import CBFFilter

        safety_registry.register("cbf", CBFFilter)
    except ImportError:
        pass

    # ----- io (ROS2) -------------------------------------------------------
    try:
        from genedynamics.deploy.io.ros2_io import ROS2RobotIO

        io_registry.register("ros2", ROS2RobotIO)
    except ImportError:
        pass

    # ----- tasks ----------------------------------------------------------
    try:
        from genedynamics.deploy.tasks.base import BaseExecutionTask
        from genedynamics.deploy.tasks.corridor_follow import CorridorFollowTask

        task_registry.register("noop", BaseExecutionTask)
        task_registry.register("corridor_follow", CorridorFollowTask)
    except ImportError:
        pass
    try:
        from genedynamics.deploy.tasks.teleop_task import TeleopTask

        task_registry.register("teleop", TeleopTask)
    except ImportError:
        pass

    # ----- observers ------------------------------------------------------
    try:
        from genedynamics.deploy.observers.logger import LoggerObserver
        from genedynamics.deploy.observers.recorder import RecorderObserver

        observer_registry.register("logger", LoggerObserver)
        observer_registry.register("recorder", RecorderObserver)
    except ImportError:
        pass
    try:
        from genedynamics.deploy.observers.ros2_publisher import ROS2PublisherObserver

        observer_registry.register("ros2_publisher", ROS2PublisherObserver)
    except ImportError:
        pass

    # ----- safety (teleop) ------------------------------------------------
    try:
        from genedynamics.deploy.safety.workspace_filter import WorkspaceFilter
        from genedynamics.deploy.safety.singularity_filter import SingularityFilter

        safety_registry.register("workspace", WorkspaceFilter)
        safety_registry.register("singularity", SingularityFilter)
    except ImportError:
        pass

    _DEFAULTS_REGISTERED = True


# Eagerly populate the registries on first import. Anyone who imports
# `genedynamics.deploy.registries` immediately gets the canonical key set.
register_defaults()
