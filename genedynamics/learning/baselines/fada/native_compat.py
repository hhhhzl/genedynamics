"""Explicit tensor-view compatibility for the pinned official mjwarp backend.

This does not change physics, labels, optimizer or model. The official DAgger
snapshot assumes Tensor.detach(); the released GPU DOF view provides the same
tensor through [:], clone() and write-through copy_(), but lacks detach().
"""

from __future__ import annotations


def install_mjwarp_snapshot_compatibility():
    from holosoma.simulator.mujoco.mjw_views import MjwDofStateView

    if hasattr(MjwDofStateView, "detach"):
        return []

    def detach(self):
        return self[:].detach()

    MjwDofStateView.detach = detach
    return ["MjwDofStateView.detach -> materialized public DOF tensor.detach"]
