"""Reference-governor interfaces (industrial, reusable across tasks).

Layering (see ``genedynamics/deploy/governor/``):

* :class:`AdmissibleSet`     -- task-specific R_adm: a reference frame + a
  safety box. This is the *only* piece a new task must implement.
* :class:`ReferenceGovernor` -- generic Step-2 projection (proximal + box
  clip + optional error throttle). Reused unchanged across tasks.

A governor sits between a follower's :class:`Intent` and the controller, in the
deploy runner loop::

    intent = follower.step(t, state)
    intent = governor.govern(intent, state, ctx)   # <-- here (runner.py)
    cmd    = controller.act(state, intent)

It never replans and never touches 2GO; worst case it is conservative (slows
the reference), which is a safety-favourable failure mode.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class Frame:
    """Planar task frame: ``origin`` (2,) and heading ``yaw`` (rad)."""

    origin: np.ndarray
    yaw: float


@dataclass
class Box:
    """Axis-aligned bounds on the frame reference ``[s, n, psi]``.

    Use ``+/-inf`` to mark an axis inactive.
    """

    lo: np.ndarray  # (3,)
    hi: np.ndarray  # (3,)

    @staticmethod
    def inactive() -> "Box":
        return Box(lo=np.full(3, -np.inf), hi=np.full(3, np.inf))


@dataclass
class Halfspace:
    """A linear safety half-space on the frame reference ``[s, n, psi]``::

        a . r >= b

    Used for obstacle-aware safety where an axis-aligned :class:`Box` is too
    coarse (e.g. avoiding round / off-axis obstacles with the body SDF). ``a``
    is the (3,) normal (typically zero on the yaw axis) and ``b`` the offset.
    """

    a: np.ndarray  # (3,)
    b: float

    @staticmethod
    def inactive() -> "Halfspace":
        return Halfspace(a=np.zeros(3), b=-np.inf)

    @property
    def active(self) -> bool:
        return bool(np.any(self.a != 0.0)) and np.isfinite(self.b)


@dataclass
class GovernorDiagnostics:
    """Per-step telemetry (mirrors 2GO's diagnostic logging style)."""

    governed: np.ndarray   # [x, y, yaw] world, the governed reference
    reference: np.ndarray  # [x, y, yaw] world, the input plan reference
    tracking_error: float  # ||x_meas - r_prev|| (m)
    rate_active: bool
    safety_active: bool
    posture_active: bool
    infeasible: bool       # safety∩rate∩posture was empty (safety prioritised)
    throttle: float        # 1.0 = no throttle (feedforward)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CorridorContext:
    """Local corridor geometry at the current reference, supplied by the caller.

    Populated by the humanoid follower (it already computes corridor width /
    centerline in its envelope). If ``half_width is None`` the safety
    constraint is inactive and the governor falls back to rate + posture only.
    """

    center: Optional[np.ndarray] = None      # (2,) centerline point (world)
    tangent_yaw: Optional[float] = None      # corridor direction (rad)
    half_width: Optional[float] = None       # corridor half-width (m)

    @classmethod
    def from_intent(cls, intent: Any) -> "CorridorContext":
        """Read a corridor context out of ``intent.extras['corridor']`` if present."""
        extras = getattr(intent, "extras", None) or {}
        c = extras.get("corridor")
        if c is None:
            return cls()
        center = c.get("center")
        return cls(
            center=None if center is None else np.asarray(center, dtype=float),
            tangent_yaw=c.get("tangent_yaw"),
            half_width=c.get("half_width"),
        )


class AdmissibleSet(ABC):
    """Task-specific feasible reference set R_adm."""

    @abstractmethod
    def reference_frame(self, ctx: Any) -> Frame:
        """Frame in which the reference ``[s, n, psi]`` is expressed."""

    @abstractmethod
    def safety_box(self, r_ref_frame: np.ndarray, ctx: Any, cfg: Any) -> Box:
        """Tightened safety bounds (already includes m_track / body half-width)."""

    def safety_halfspace(self, r_ref_frame: np.ndarray, ctx: Any, cfg: Any) -> "Halfspace":
        """Optional linearised safety half-space ``a . r >= b`` in frame coords.

        Default: inactive (box-only sets need not implement it). Obstacle-aware
        sets override this to push the reference away from obstacles.
        """
        return Halfspace.inactive()


class ReferenceGovernor(ABC):
    """Generic Step-2 governor."""

    @abstractmethod
    def reset(self, state: Any = None) -> None:
        """Re-anchor the previous governed reference (call at episode start)."""

    @abstractmethod
    def govern(self, ref: Any, state: Any, ctx: Any = None) -> Any:
        """Map a plan reference into an execution-compatible governed reference."""
