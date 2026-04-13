"""Cross-runtime array conversion bridge.

Provides transparent conversion between numpy, jax, and torch array types
so that an IO producing one runtime's arrays can be paired with a controller
expecting another. Conversions are lazy-imported to avoid pulling in heavy
dependencies when they are not needed.

Conversion table
----------------

===============  ==========================================
source → target  implementation
===============  ==========================================
numpy → jax      ``jnp.asarray(x)``
jax → numpy      ``np.asarray(jax.device_get(x))``
numpy → torch    ``torch.from_numpy(x.copy())``
torch → numpy    ``x.detach().cpu().numpy()``
jax → torch      jax → numpy → torch (no dlpack in v1)
torch → jax      torch → numpy → jax (no dlpack in v1)
same → same      identity (no copy)
===============  ==========================================

The :class:`ArrayBridge` wraps a :class:`RobotIO` so that ``get_state``
returns arrays in the *target* runtime and ``send_control`` accepts arrays
in the target runtime, converting them back to the IO's native runtime
before dispatch.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

import numpy as np

from genedynamics.deploy.interfaces.messages import ControlCommand, RobotState

__all__ = ["convert", "ArrayBridge"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy module handles — populated on first use
# ---------------------------------------------------------------------------

_jax = None
_jnp = None
_torch = None


def _ensure_jax():
    global _jax, _jnp
    if _jax is None:
        import jax
        import jax.numpy as jnp

        _jax = jax
        _jnp = jnp


def _ensure_torch():
    global _torch
    if _torch is None:
        import torch

        _torch = torch


# ---------------------------------------------------------------------------
# Single-array conversion
# ---------------------------------------------------------------------------


def convert(arr: Any, target_runtime: str) -> Any:
    """Convert *arr* to the array type used by *target_runtime*.

    Supported runtimes: ``"numpy"``, ``"jax"``, ``"torch"``.

    If *arr* is already the right type (or is ``None``), it is returned
    unchanged with no copy.
    """
    if arr is None:
        return None

    source = _detect_runtime(arr)
    if source == target_runtime:
        return arr

    # Route through numpy as the universal intermediate.
    as_np = _to_numpy(arr, source)
    return _from_numpy(as_np, target_runtime)


def _detect_runtime(arr: Any) -> str:
    """Return the runtime tag for *arr*."""
    type_name = type(arr).__module__
    if isinstance(arr, np.ndarray):
        return "numpy"
    if "jax" in type_name:
        return "jax"
    if "torch" in type_name:
        return "torch"
    # Fall back to numpy for scalars / plain Python objects.
    return "numpy"


def _to_numpy(arr: Any, source: str) -> np.ndarray:
    if source == "numpy":
        return np.asarray(arr)
    if source == "jax":
        _ensure_jax()
        return np.asarray(_jax.device_get(arr))
    if source == "torch":
        return arr.detach().cpu().numpy()
    raise ValueError(f"Unsupported source runtime: {source!r}")


def _from_numpy(arr: np.ndarray, target: str) -> Any:
    if target == "numpy":
        return arr
    if target == "jax":
        _ensure_jax()
        return _jnp.asarray(arr)
    if target == "torch":
        _ensure_torch()
        return _torch.from_numpy(arr.copy())
    raise ValueError(f"Unsupported target runtime: {target!r}")


# ---------------------------------------------------------------------------
# RobotState / ControlCommand field walkers
# ---------------------------------------------------------------------------

# Explicit field lists — no dataclasses.fields() magic so we keep full
# control over which fields are array-typed and which are metadata.

_ROBOT_STATE_ARRAY_FIELDS = (
    "qpos",
    "qvel",
    "base_pose",
    "base_twist",
    "joint_torque",
)

_CONTROL_COMMAND_ARRAY_FIELDS = (
    "joint_pos",
    "joint_vel",
    "joint_torque",
    "kp",
    "kd",
)


def _convert_state(state: RobotState, target: str) -> RobotState:
    """Return a shallow copy of *state* with array fields converted."""
    kwargs: dict[str, Any] = {"t": state.t, "extras": state.extras}
    for name in _ROBOT_STATE_ARRAY_FIELDS:
        kwargs[name] = convert(getattr(state, name), target)
    # contact is a mapping of str → array; walk each value.
    if state.contact is not None:
        kwargs["contact"] = {
            k: convert(v, target) for k, v in state.contact.items()
        }
    else:
        kwargs["contact"] = None
    # imu is kept as-is (raw sensor data, typically small numpy arrays).
    kwargs["imu"] = state.imu
    return RobotState(**kwargs)


def _convert_command(cmd: ControlCommand, target: str) -> ControlCommand:
    """Return a shallow copy of *cmd* with array fields converted."""
    kwargs: dict[str, Any] = {
        "kind": cmd.kind,
        "loco_cmd": cmd.loco_cmd,
        "extras": cmd.extras,
    }
    for name in _CONTROL_COMMAND_ARRAY_FIELDS:
        kwargs[name] = convert(getattr(cmd, name), target)
    return ControlCommand(**kwargs)


# ---------------------------------------------------------------------------
# ArrayBridge — wraps a RobotIO for transparent runtime conversion
# ---------------------------------------------------------------------------


class ArrayBridge:
    """Wrap a :class:`RobotIO` so its outputs/inputs are transparently converted.

    Composition pattern — the bridge delegates every call to the wrapped IO
    and converts arrays at the boundary::

        bridge = ArrayBridge(io, target_runtime="numpy")
        state  = bridge.get_state()       # → RobotState with numpy arrays
        bridge.send_control(cmd)           # numpy arrays → io.array_runtime

    The bridge owns no mutable state beyond the reference to the inner IO.
    All protocol-required attributes (``spec``, ``physics_backend``,
    ``accepts``) are forwarded from the wrapped IO. ``array_runtime`` is
    overridden to reflect the *target* runtime.
    """

    def __init__(self, io: Any, target_runtime: str) -> None:
        self._io = io
        self._target = target_runtime
        self._source = io.array_runtime
        logger.info(
            "ArrayBridge: wrapping %s (%s) → target runtime %s",
            type(io).__name__,
            self._source,
            self._target,
        )

    # Forward protocol attributes.
    @property
    def spec(self) -> Any:
        return self._io.spec

    @property
    def physics_backend(self) -> Optional[str]:
        return self._io.physics_backend

    @property
    def array_runtime(self) -> str:
        return self._target

    @property
    def accepts(self) -> tuple[str, ...]:
        return self._io.accepts

    # Episode management.

    def reset(self, episode_id: Optional[str] = None) -> RobotState:
        state = self._io.reset(episode_id)
        return _convert_state(state, self._target)

    def get_state(self) -> RobotState:
        state = self._io.get_state()
        return _convert_state(state, self._target)

    def send_control(self, cmd: ControlCommand) -> None:
        converted = _convert_command(cmd, self._source)
        self._io.send_control(converted)

    def step(self, dt: float) -> RobotState:
        state = self._io.step(dt)
        return _convert_state(state, self._target)

    def close(self) -> None:
        self._io.close()

    # Forward any extra attribute (physics queries, etc.) to the inner IO.
    def __getattr__(self, name: str) -> Any:
        return getattr(self._io, name)
