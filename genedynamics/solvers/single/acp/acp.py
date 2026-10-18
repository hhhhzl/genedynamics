"""ACP solver frontend.

The frontend follows the same package contract as the other single solvers:
it owns lifecycle and delegates numerical work to a backend implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


class ACPSolver:
    """Run a native ACP policy through the selected numerical backend."""

    def __init__(self, policy: Any, shape_meta: Mapping[str, Any], **backend_options: Any):
        from .backends.acp_torch import ACPBackendTorch

        self._backend_impl = ACPBackendTorch(policy, shape_meta, **backend_options)

    @property
    def backend(self):
        return self._backend_impl

    @property
    def policy(self):
        return self._backend_impl.policy

    @property
    def shape_meta(self):
        return self._backend_impl.shape_meta

    @property
    def device(self):
        return self._backend_impl.device

    @property
    def execution_horizon(self):
        return self._backend_impl.execution_horizon

    @property
    def sample_interval_s(self):
        return self._backend_impl.sample_interval_s

    def reset(self) -> None:
        self._backend_impl.reset()

    def predict_action(self, sparse: Mapping[str, Any], current_tool_pose_world: Any):
        return self._backend_impl.predict_action(sparse, current_tool_pose_world)

    def act(self, public_observation: Mapping[str, Any]):
        return self._backend_impl.act(public_observation)

    def load_policy_state_dict(self, payload: Mapping[str, Any], *, use_ema: bool = True) -> None:
        self._backend_impl.load_policy_state_dict(payload, use_ema=use_ema)

    def load_checkpoint(self, path: str | Path, *, use_ema: bool = True) -> None:
        self._backend_impl.load_checkpoint(path, use_ema=use_ema)


__all__ = ["ACPSolver"]
