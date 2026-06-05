"""Corridor admissible set: the R_adm safety component for humanoid corridors.

The reference frame is aligned with the corridor centerline (s = longitudinal,
n = lateral). The only safety constraint is a lateral-clearance interval,
tightened by ``m_track`` (tracking margin) and the body half-width:

    |n| <= half_width - body_half_width - m_track

Longitudinal and yaw axes are left to the generic rate / posture boxes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import AdmissibleSet, Box, CorridorContext, Frame


class CorridorAdmissibleSet(AdmissibleSet):
    def _ctx(self, ctx: Any) -> CorridorContext:
        if isinstance(ctx, CorridorContext):
            return ctx
        if ctx is None:
            return CorridorContext()
        # allow a plain Intent or dict to be passed through
        if hasattr(ctx, "extras"):
            return CorridorContext.from_intent(ctx)
        if isinstance(ctx, dict):
            return CorridorContext(
                center=None if ctx.get("center") is None else np.asarray(ctx["center"], float),
                tangent_yaw=ctx.get("tangent_yaw"),
                half_width=ctx.get("half_width"),
            )
        return CorridorContext()

    def reference_frame(self, ctx: Any) -> Frame:
        c = self._ctx(ctx)
        if c.center is None or c.tangent_yaw is None:
            return Frame(origin=np.zeros(2), yaw=0.0)  # identity (world) frame
        return Frame(origin=np.asarray(c.center, dtype=float), yaw=float(c.tangent_yaw))

    def safety_box(self, r_ref_frame: np.ndarray, ctx: Any, cfg: Any) -> Box:
        c = self._ctx(ctx)
        if c.half_width is None:
            return Box.inactive()
        n_clear = float(c.half_width) - float(cfg.body_half_width) - float(cfg.m_track)
        lo = np.array([-np.inf, -n_clear, -np.inf])
        hi = np.array([np.inf, n_clear, np.inf])
        return Box(lo=lo, hi=hi)
