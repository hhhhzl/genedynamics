"""Runtime compatibility validation for built deploy components.

Called by :func:`~genedynamics.deploy.config_schema.build_components` after
all components are instantiated. The validator either:

1. confirms that ``io``, ``controller``, and ``safety`` share compatible
   ``array_runtime`` / ``runtime`` tags, **or**
2. wraps the IO in an :class:`~genedynamics.deploy.array_bridge.ArrayBridge`
   when automatic conversion is possible, **or**
3. raises :class:`RuntimeMismatchError` with a clear diagnostic message.

The ``"any"`` runtime sentinel is recognised for safety filters that operate
purely on :class:`ControlCommand` metadata and are agnostic to the array
backend.
"""

from __future__ import annotations

import logging
from typing import Optional

from genedynamics.deploy.config_schema import BuiltComponents

__all__ = ["validate_runtimes", "RuntimeMismatchError", "CommandKindError"]

logger = logging.getLogger(__name__)

# Runtimes for which ArrayBridge can transparently convert.
_BRIDGEABLE = frozenset({"numpy", "jax", "torch"})

# Sentinel: safety filters may declare this to accept any runtime.
RUNTIME_ANY = "any"


class CommandKindError(Exception):
    """Raised when a controller produces command kinds the IO does not accept.

    Attributes:
        produces: The controller's ``produces`` tuple.
        accepts: The IO's ``accepts`` tuple.
        unsupported: The set of kinds produced but not accepted.
    """

    def __init__(
        self,
        produces: tuple[str, ...],
        accepts: tuple[str, ...],
        unsupported: set[str],
    ) -> None:
        self.produces = produces
        self.accepts = accepts
        self.unsupported = unsupported
        super().__init__(
            f"Controller produces command kinds {sorted(unsupported)} "
            f"that the IO does not accept. "
            f"Controller.produces={sorted(produces)}, "
            f"IO.accepts={sorted(accepts)}."
        )


class RuntimeMismatchError(Exception):
    """Raised when IO / controller / safety runtimes are incompatible.

    Attributes:
        io_runtime: The IO's ``array_runtime`` tag.
        controller_runtime: The controller's ``runtime`` tag.
        safety_runtime: The safety filter's ``runtime`` tag (or ``None``).
    """

    def __init__(
        self,
        io_runtime: str,
        controller_runtime: str,
        safety_runtime: Optional[str],
    ) -> None:
        self.io_runtime = io_runtime
        self.controller_runtime = controller_runtime
        self.safety_runtime = safety_runtime
        super().__init__(
            f"Runtime mismatch: io={io_runtime!r}, "
            f"controller={controller_runtime!r}, "
            f"safety={safety_runtime!r}. "
            f"Bridgeable runtimes are {sorted(_BRIDGEABLE)}. "
            f"Set DeployConfig.strict_runtime = True to fail on any mismatch, "
            f"or ensure all components share a compatible runtime."
        )


def _get_runtime(component, attr: str) -> Optional[str]:
    """Read a runtime tag from a component, returning ``None`` if absent."""
    return getattr(component, attr, None)


def validate_runtimes(
    built: BuiltComponents,
    *,
    strict: bool = False,
) -> BuiltComponents:
    """Validate or bridge the runtime tags across IO, controller, and safety.

    Args:
        built: The assembled components from :func:`build_components`.
        strict: When ``True``, raise :class:`RuntimeMismatchError` on *any*
            mismatch instead of attempting automatic bridging. Defaults to
            ``False`` (bridge and log).

    Returns:
        The same :class:`BuiltComponents`, possibly with ``built.io``
        replaced by an :class:`ArrayBridge` wrapper.

    Raises:
        RuntimeMismatchError: When the runtime combination cannot be
            resolved (or when *strict* is ``True`` and a mismatch exists).
    """
    if built.io is None or built.controller is None:
        return built

    # Validate command kinds: controller.produces ⊆ io.accepts.
    _validate_command_kinds(built.io, built.controller)

    io_rt = _get_runtime(built.io, "array_runtime") or "numpy"
    ctl_rt = _get_runtime(built.controller, "runtime") or "numpy"
    safety_rt = _get_runtime(built.safety, "runtime") if built.safety is not None else None

    # 1. All compatible — nothing to do.
    if _all_compatible(io_rt, ctl_rt, safety_rt):
        logger.debug(
            "Runtime check: all components compatible (io=%s, ctl=%s, safety=%s)",
            io_rt, ctl_rt, safety_rt,
        )
        return built

    # 2. Strict mode — reject any mismatch.
    if strict:
        raise RuntimeMismatchError(io_rt, ctl_rt, safety_rt)

    # 3. Try bridging: wrap IO to match the controller's runtime.
    if io_rt in _BRIDGEABLE and ctl_rt in _BRIDGEABLE:
        from genedynamics.deploy.array_bridge import ArrayBridge

        logger.info(
            "RuntimeBridge: wrapping %s (%s) for %s (%s)",
            type(built.io).__name__,
            io_rt,
            type(built.controller).__name__,
            ctl_rt,
        )
        built.io = ArrayBridge(built.io, target_runtime=ctl_rt)
        return built

    # 4. Cannot bridge — raise.
    raise RuntimeMismatchError(io_rt, ctl_rt, safety_rt)


def _validate_command_kinds(io: Any, controller: Any) -> None:
    """Raise :class:`CommandKindError` if controller produces unsupported kinds."""
    io_accepts = getattr(io, "accepts", None)
    ctl_produces = getattr(controller, "produces", None)
    if not io_accepts or not ctl_produces:
        return  # No metadata to validate.
    accepts_set = set(io_accepts)
    produces_set = set(ctl_produces)
    unsupported = produces_set - accepts_set
    if unsupported:
        raise CommandKindError(ctl_produces, io_accepts, unsupported)
    logger.debug(
        "Command kind check: controller.produces=%s ⊆ io.accepts=%s",
        sorted(produces_set), sorted(accepts_set),
    )


def _all_compatible(
    io_rt: str,
    ctl_rt: str,
    safety_rt: Optional[str],
) -> bool:
    """Return ``True`` if all three runtimes are mutually compatible."""
    # io and controller must match.
    if io_rt != ctl_rt:
        return False
    # Safety is compatible if it matches, is "any", or is absent.
    if safety_rt is None or safety_rt == RUNTIME_ANY or safety_rt == ctl_rt:
        return True
    return False
