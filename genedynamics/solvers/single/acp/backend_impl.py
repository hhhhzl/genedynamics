"""Backend contract for the ACP solver frontend."""

from __future__ import annotations

from typing import Any, Mapping, Protocol


class ACPBackend(Protocol):
    """Runtime surface implemented by an ACP numerical backend."""

    @property
    def policy(self) -> Any: ...

    def reset(self) -> None: ...

    def predict_action(self, sparse: Mapping[str, Any], current_tool_pose_world: Any) -> Mapping[str, Any]: ...

    def load_policy_state_dict(self, payload: Mapping[str, Any], *, use_ema: bool = True) -> None: ...


__all__ = ["ACPBackend"]
