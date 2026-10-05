"""Nested-class config schema and registry-driven initializer.

This module is the **new** Phase 9 config layer. It coexists with the
legacy ``deploy/config.py`` (which the old CLI still consumes); the new
:class:`DeployConfig` here is what every preset under ``deploy/presets/``
inherits from. Migration is incremental — both code paths are valid for
now, the legacy path can be removed once every entry point flips over.

The single most important helper is :func:`initialize_class`, which walks
a config node, looks up its ``registry_key`` in the appropriate registry
(see :mod:`genedynamics.deploy.registries`), gathers its constructor
kwargs from the rest of the node, and instantiates the class. Nested
config classes are recursively flattened to dicts so a controller's
``config`` sub-class is passed as a single ``config`` kwarg the
controller can consume directly.

Why nested classes instead of dicts? They give:

* IDE autocomplete and type checking on every config field
* Inheritance — a real-G1 preset can subclass the sim preset and override
  one inner class without copy-pasting 200 lines
* Free namespacing — ``cfg.controller.config.task_weights`` reads as it's
  laid out in code

Spark uses essentially the same pattern (``init_member_classes``); we
deviate by going through ``core/registry/`` instead of module-name
reflection so that swapping implementations doesn't require renaming or
moving Python modules.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from genedynamics.deploy.registries import get_registry, resolve_class

logger = logging.getLogger(__name__)

__all__ = [
    "DeployConfig",
    "ComponentConfig",
    "Lazy",
    "config_to_dict",
    "initialize_class",
    "build_components",
    "BuiltComponents",
]


class Lazy:
    """Marker wrapping a zero-arg factory.

    Use this in a preset when a field's value depends on optional imports
    (ROS2, Unitree SDK, …). The factory is called once when
    :func:`initialize_class` resolves the node, so the preset module
    itself stays import-safe on every machine::

        class io(ComponentConfig):
            registry_key = "io.unitree_g1"
            localization = Lazy(_build_localization_plugin)
    """

    __slots__ = ("factory",)

    def __init__(self, factory):
        if not callable(factory):
            raise TypeError(f"Lazy expects a callable, got {type(factory).__name__}")
        self.factory = factory

    def resolve(self):
        return self.factory()


# ---------------------------------------------------------------------------
# Config base classes
# ---------------------------------------------------------------------------


class ComponentConfig:
    """Marker base for any nested-class config node.

    A node carries a ``registry_key`` (e.g. ``"controller.wbc"``) plus
    arbitrary attributes / nested ``ComponentConfig`` subclasses for the
    component's constructor kwargs.

    Subclassing :class:`ComponentConfig` is **optional** — the helpers in
    this module accept any class with class-level attributes — but it
    makes intent obvious to readers.
    """


class DeployConfig:
    """Root config node for a deploy run.

    Conventional fields (subclasses override or add as needed):

    * ``control_hz``     — control loop frequency
    * ``sim_dt``         — physics step (sim only)
    * ``max_steps``      — episode length cap
    * ``runtime``        — nested ``ComponentConfig`` selecting the runtime
      backend (``name = "numpy"`` / ``"jax"`` / ``"torch"``).
    * ``robot``          — nested ``ComponentConfig`` with ``id`` matching
      :mod:`genedynamics.robots.registry`.
    * ``io``             — nested config with ``registry_key = "io.<x>"``.
    * ``follower``       — nested config with ``registry_key = "follower.<x>"``.
    * ``controller``     — nested config with ``registry_key = "controller.<x>"``.
    * ``safety``         — nested config with ``registry_key = "safety.<x>"``.
    * ``task``           — nested config with ``registry_key = "task.<x>"``.
    * ``observers``      — list of dicts (each with a ``registry_key``).

    Subclasses don't need to define every field — only what differs from
    the parent. The :func:`build_components` helper tolerates missing
    sections (returns ``None`` for the corresponding component).
    """

    control_hz: float = 200.0
    sim_dt: float = 1.0 / 1000.0
    max_steps: int = 20_000


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def config_to_dict(node: Any) -> Dict[str, Any]:
    """Recursively convert a nested-class config node to a plain dict.

    Walks class-level attributes (skipping dunders and methods), converting
    nested classes to nested dicts. Lists / tuples / dicts are walked
    element-wise. Anything else is passed through as-is.
    """
    if node is None:
        return {}
    if isinstance(node, dict):
        return {k: _convert_value(v) for k, v in node.items()}
    out: Dict[str, Any] = {}
    seen: set[str] = set()
    # Walk MRO so subclass overrides win.
    cls = node if inspect.isclass(node) else type(node)
    for klass in cls.__mro__:
        if klass is object:
            continue
        for name, value in vars(klass).items():
            if name.startswith("_") or name in seen:
                continue
            if callable(value) and not inspect.isclass(value):
                continue
            seen.add(name)
            out[name] = _convert_value(value)
    return out


def _convert_value(value: Any) -> Any:
    if isinstance(value, Lazy):
        return value
    if inspect.isclass(value):
        return config_to_dict(value)
    if isinstance(value, dict):
        return {k: _convert_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        converted = [_convert_value(v) for v in value]
        return converted if isinstance(value, list) else tuple(converted)
    return value


# ---------------------------------------------------------------------------
# initialize_class — registry-driven instantiation
# ---------------------------------------------------------------------------


def initialize_class(
    node: Any,
    *,
    extra_kwargs: Optional[Mapping[str, Any]] = None,
    nested_extra_kwargs: Optional[Mapping[str, Any]] = None,
    nested_config_kw: str = "config",
) -> Any:
    """Instantiate a registered component from a nested-class config node.

    Steps:

    1. Convert ``node`` to a dict via :func:`config_to_dict`.
    2. Pop ``registry_key`` (mandatory) and resolve it via
       :mod:`genedynamics.deploy.registries`.
    3. Filter the remaining keys against the resolved class's
       constructor signature. Keys that match a constructor parameter are
       passed as-is; nested-dict values are kept as dicts (the constructor
       gets to decode them).
    4. If the class has a ``config`` parameter and the node has any
       remaining nested-dict keys that don't match other constructor
       parameters, those keys are bundled under a ``config`` kwarg so
       components like the WBC controller can take a single grouped
       config object.
    5. Merge ``extra_kwargs`` (which take precedence — used for runtime
       cross-cutting deps like ``spec``, ``runtime``, ``robot``).
    6. Instantiate the class with the assembled kwargs.

    Returns the instantiated component.
    """
    cfg_dict = config_to_dict(node)
    if "registry_key" not in cfg_dict:
        raise KeyError(
            "initialize_class: missing 'registry_key' on config node "
            f"{type(node).__name__}"
        )
    key = cfg_dict.pop("registry_key")
    cls = resolve_class(key)

    # Inspect the resolved class's __init__ to filter kwargs.
    try:
        sig = inspect.signature(cls.__init__)
        accepted = {
            name
            for name in sig.parameters
            if name != "self"
        }
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
    except (TypeError, ValueError):  # pragma: no cover - C extensions
        accepted = set(cfg_dict)
        accepts_kwargs = True

    direct_kwargs: Dict[str, Any] = {}
    leftover: Dict[str, Any] = {}
    for k, v in cfg_dict.items():
        if isinstance(v, Lazy):
            v = v.resolve()
        elif isinstance(v, dict) and "registry_key" in v:
            # Nested component nodes (for example a sport controller's
            # locomotion client) are components, not opaque config dicts.
            v = _initialize_dict(dict(v), nested_extra_kwargs)
        if k in accepted or accepts_kwargs:
            direct_kwargs[k] = v
        else:
            leftover[k] = v
    if leftover and nested_config_kw in accepted and nested_config_kw not in direct_kwargs:
        direct_kwargs[nested_config_kw] = leftover

    if extra_kwargs:
        direct_kwargs.update(extra_kwargs)

    try:
        return cls(**direct_kwargs)
    except TypeError as exc:
        raise TypeError(
            f"initialize_class({key!r}) failed instantiating {cls.__name__}: {exc}\n"
            f"  candidate kwargs: {sorted(direct_kwargs)}"
        ) from exc


# ---------------------------------------------------------------------------
# Whole-pipeline assembly
# ---------------------------------------------------------------------------


@dataclass
class BuiltComponents:
    """Container for the components a :class:`DeployConfig` resolves to."""

    runtime: Optional[Any] = None
    robot: Optional[Any] = None
    io: Optional[Any] = None
    follower: Optional[Any] = None
    controller: Optional[Any] = None
    safety: Optional[Any] = None
    task: Optional[Any] = None
    observers: List[Any] = None  # type: ignore[assignment]


def build_components(
    cfg: type[DeployConfig] | DeployConfig,
    *,
    strict: bool = False,
) -> BuiltComponents:
    """Walk a :class:`DeployConfig` and instantiate every declared component.

    The build order is:
    ``runtime → robot → io → follower → controller → safety → task → observers``.
    Each subsequent component receives the spec / runtime / io of the
    previous as ``extra_kwargs`` so the preset doesn't have to repeat
    them. Sections that don't appear on the config are skipped (returned
    as ``None``); the pipeline runner is responsible for treating their
    absence as a config error if it cares.

    Args:
        cfg: The deploy config (class or instance).
        strict: When ``True``, raise if ``io`` or ``controller`` sections
            are missing on the config. Defaults to ``False`` for backwards
            compatibility.

    The function never opens a network socket or starts a sim — it just
    constructs the objects. The pipeline loop in
    :mod:`genedynamics.deploy.runner` is what actually runs them.
    """
    built = BuiltComponents(observers=[])

    runtime_node = _maybe_node(cfg, "runtime")
    if runtime_node is not None:
        built.runtime = _build_runtime(runtime_node)

    robot_node = _maybe_node(cfg, "robot")
    if robot_node is not None:
        built.robot = _build_robot(robot_node)

    io_node = _maybe_node(cfg, "io")
    if io_node is not None:
        built.io = initialize_class(io_node)

    extras: Dict[str, Any] = {}
    if built.io is not None and hasattr(built.io, "spec"):
        extras["spec"] = built.io.spec

    follower_node = _maybe_node(cfg, "follower")
    if follower_node is not None:
        built.follower = initialize_class(follower_node, extra_kwargs=_only_accepted(follower_node, extras))

    controller_node = _maybe_node(cfg, "controller")
    if controller_node is not None:
        ctl_extras = dict(extras)
        if built.io is not None:
            ctl_extras["io"] = built.io
        nested_extras = dict(ctl_extras)
        if built.io is not None:
            # Locomotion policies need an object exposing the active robot
            # spec; every concrete IO provides that contract.
            nested_extras["robot"] = built.io
        built.controller = initialize_class(
            controller_node,
            extra_kwargs=_only_accepted(controller_node, ctl_extras),
            nested_extra_kwargs=nested_extras,
        )

    safety_node = _maybe_node(cfg, "safety")
    if safety_node is not None:
        built.safety = _build_safety(safety_node, extras)

    task_node = _maybe_node(cfg, "task")
    if task_node is not None:
        built.task = initialize_class(task_node, extra_kwargs=_only_accepted(task_node, extras))

    observers_field = getattr(cfg, "observers", None)
    if observers_field:
        for entry in observers_field:
            if isinstance(entry, dict):
                node_dict = dict(entry)
            else:
                node_dict = config_to_dict(entry)
            built.observers.append(_initialize_dict(node_dict))

    # Strict build validation: require io + controller.
    build_strict = strict or bool(getattr(cfg, "strict_build", False))
    if build_strict:
        missing = []
        if built.io is None:
            missing.append("io")
        if built.controller is None:
            missing.append("controller")
        if missing:
            raise RuntimeError(
                f"build_components(strict=True): required sections missing "
                f"on {type(cfg).__name__}: {missing}"
            )

    from genedynamics.deploy.runtime_check import validate_runtimes

    runtime_strict = bool(getattr(cfg, "strict_runtime", False))
    return validate_runtimes(built, strict=runtime_strict)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _maybe_node(cfg: Any, name: str) -> Optional[Any]:
    return getattr(cfg, name, None)


def _build_runtime(node: Any) -> Any:
    """Resolve a runtime backend via :class:`RuntimeBackendManager`."""
    cfg_dict = config_to_dict(node)
    name = cfg_dict.pop("name", None)
    if name is None:
        return None
    try:
        from genedynamics.core.backends.runtime import RuntimeBackendManager

        return RuntimeBackendManager.set_backend(name, **cfg_dict)
    except ImportError:
        return None


def _build_robot(node: Any) -> Any:
    """Resolve a :class:`RobotEntry` from :mod:`genedynamics.robots.registry`.

    Accepts either a single ``id`` (legacy single-key) split on ``.`` into
    ``(robot_type, model_id)``, or the explicit ``robot_type`` + ``model_id``
    pair. Returns the :class:`RobotEntry` (metadata only), not a constructed
    robot — building the kinematic model is the IO's job and depends on
    physics backend.
    """
    cfg_dict = config_to_dict(node)
    rid = cfg_dict.pop("id", None)
    robot_type = cfg_dict.pop("robot_type", None)
    model_id = cfg_dict.pop("model_id", None)
    if rid is not None and robot_type is None:
        if "." in rid:
            robot_type, model_id = rid.split(".", 1)
        else:
            robot_type, model_id = "humanoid", rid
    if robot_type is None or model_id is None:
        return None
    try:
        from genedynamics.robots.registry import get_robot_registry

        return get_robot_registry().get(robot_type, model_id)
    except ImportError:
        logger.warning(
            "Robot registry unavailable (genedynamics.robots not installed); "
            "skipping robot resolution for %s/%s", robot_type, model_id,
        )
        return None
    except Exception as exc:
        logger.error(
            "Failed to resolve robot %s/%s: %s", robot_type, model_id, exc,
            exc_info=True,
        )
        raise


def _build_safety(node: Any, extras: Mapping[str, Any]) -> Any:
    """Build a (possibly composite) safety filter.

    Recognizes two forms:

    1. ``registry_key = "safety.composite"`` with a ``filters`` list of
       sub-config nodes / dicts → constructs each child via
       ``initialize_class`` then wraps them in a
       :class:`CompositeSafetyFilter`.
    2. Any other ``safety.<x>`` → constructed directly with the spec
       extra_kwargs.
    """
    cfg_dict = config_to_dict(node)
    key = cfg_dict.get("registry_key")
    if key == "safety.composite":
        children_specs = cfg_dict.get("filters", []) or []
        children: List[Any] = []
        for entry in children_specs:
            if isinstance(entry, str):
                children.append(_initialize_dict({"registry_key": entry}, extras))
            else:
                child_dict = entry if isinstance(entry, dict) else config_to_dict(entry)
                children.append(_initialize_dict(child_dict, extras))
        from genedynamics.deploy.safety.composite import CompositeSafetyFilter

        return CompositeSafetyFilter(spec=extras.get("spec"), filters=children)
    return initialize_class(node, extra_kwargs=_only_accepted(node, extras))


def _initialize_dict(cfg_dict: Dict[str, Any], extras: Optional[Mapping[str, Any]] = None) -> Any:
    """Same as :func:`initialize_class` but starting from a dict."""
    if "registry_key" not in cfg_dict:
        raise KeyError(f"missing 'registry_key' in config dict: {cfg_dict}")
    cls = resolve_class(cfg_dict["registry_key"])
    payload = {k: v for k, v in cfg_dict.items() if k != "registry_key"}
    if extras:
        # Filter extras against the constructor's accepted args.
        try:
            sig = inspect.signature(cls.__init__)
            accepted = {
                name
                for name in sig.parameters
                if name != "self"
            }
            for k, v in extras.items():
                if k in accepted and k not in payload:
                    payload[k] = v
        except (TypeError, ValueError):  # pragma: no cover
            payload.update(extras)
    return cls(**payload)


def _only_accepted(node: Any, extras: Mapping[str, Any]) -> Dict[str, Any]:
    """Trim ``extras`` down to keys the registered class actually wants."""
    cfg_dict = config_to_dict(node)
    key = cfg_dict.get("registry_key")
    if key is None:
        return dict(extras)
    try:
        cls = resolve_class(key)
        sig = inspect.signature(cls.__init__)
        accepted = {n for n in sig.parameters if n != "self"}
        return {k: v for k, v in extras.items() if k in accepted}
    except KeyError:
        logger.debug("Registry key %r not found during extras filtering; passing all extras", key)
        return dict(extras)
    except Exception as exc:
        logger.debug("Could not inspect %r for extras filtering: %s", key, exc)
        return dict(extras)
