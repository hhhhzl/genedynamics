"""State / action timeseries recorder.

Buffers raw arrays in memory during the episode and dumps them to one
``<episode_id>.npz`` per episode at :meth:`on_episode_end`. This is the
"replay-ready" companion to the JSONL :class:`LoggerObserver`.

The recorder captures, per step:

* ``t``               — sim time
* ``qpos`` / ``qvel`` — state arrays (whatever shape the IO emits)
* ``cmd_kind``        — command discriminator string
* ``cmd_joint_pos`` / ``cmd_joint_torque`` — when present
* ``intent_lin_vel`` / ``intent_yaw_rate`` — for follower diagnostics

Arrays with inconsistent shapes (e.g. controllers that switch between
``"joint_pos"`` and ``"loco"``) are right-padded with NaNs so the resulting
``.npz`` is regular. Set ``include_state=False`` to record only commands +
intents (lighter footprint for long rollouts).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Mapping, Optional

import numpy as np

from genedynamics.deploy.interfaces.messages import (
    ControlCommand,
    Intent,
    RobotState,
    StepInfo,
)
from genedynamics.deploy.observers.base import BaseObserver

__all__ = ["RecorderObserver"]


class RecorderObserver(BaseObserver):
    """In-memory rollout recorder that flushes to ``.npz`` per episode.

    Args:
        out_dir: Directory where ``<episode_id>.npz`` files are written.
        name: Observer display name.
        include_state: When ``False``, skip ``qpos`` / ``qvel`` capture
            (useful for long rollouts on memory-constrained machines).
    """

    def __init__(
        self,
        out_dir: str | Path,
        *,
        name: str = "recorder",
        include_state: bool = True,
    ) -> None:
        super().__init__(name=name)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.include_state = bool(include_state)
        self._reset_buffers()
        self._episode_id: Optional[str] = None
        self._metadata: Mapping[str, Any] = {}

    def _reset_buffers(self) -> None:
        self._t: List[float] = []
        self._qpos: List[np.ndarray] = []
        self._qvel: List[np.ndarray] = []
        self._cmd_kind: List[str] = []
        self._cmd_joint_pos: List[Optional[np.ndarray]] = []
        self._cmd_joint_torque: List[Optional[np.ndarray]] = []
        self._intent_lin_vel: List[Optional[np.ndarray]] = []
        self._intent_yaw_rate: List[float] = []
        self._reward: List[float] = []
        self._cost: List[float] = []
        self._success: List[float] = []  # 0/1/NaN

    # ------------------------------------------------------------------
    # Observer protocol
    # ------------------------------------------------------------------

    def on_episode_start(self, episode_id: str, metadata: Mapping[str, Any]) -> None:
        self._reset_buffers()
        self._episode_id = episode_id
        self._metadata = dict(metadata) if metadata else {}

    def on_step(
        self,
        t: float,
        state: RobotState,
        intent: Intent,
        cmd: ControlCommand,
        info: StepInfo,
    ) -> None:
        self._t.append(float(t))
        if self.include_state:
            self._qpos.append(np.asarray(state.qpos, dtype=np.float64).copy())
            self._qvel.append(np.asarray(state.qvel, dtype=np.float64).copy())
        self._cmd_kind.append(str(cmd.kind))
        self._cmd_joint_pos.append(
            None if cmd.joint_pos is None else np.asarray(cmd.joint_pos, dtype=np.float64).copy()
        )
        self._cmd_joint_torque.append(
            None
            if cmd.joint_torque is None
            else np.asarray(cmd.joint_torque, dtype=np.float64).copy()
        )
        if intent is not None:
            self._intent_lin_vel.append(
                np.asarray(intent.base_lin_vel, dtype=np.float64).copy()
            )
            self._intent_yaw_rate.append(float(intent.base_yaw_rate))
        else:
            self._intent_lin_vel.append(None)
            self._intent_yaw_rate.append(float("nan"))
        self._reward.append(float(info.reward) if info.reward is not None else float("nan"))
        self._cost.append(float(info.cost) if info.cost is not None else float("nan"))
        self._success.append(
            1.0 if info.success is True else (0.0 if info.success is False else float("nan"))
        )

    def on_episode_end(self, summary: Mapping[str, Any]) -> None:
        if self._episode_id is None or not self._t:
            return
        path = self.out_dir / f"{self._episode_id}.npz"
        payload = {
            "t": np.asarray(self._t, dtype=np.float64),
            "cmd_kind": np.asarray(self._cmd_kind, dtype=object),
            "intent_yaw_rate": np.asarray(self._intent_yaw_rate, dtype=np.float64),
            "reward": np.asarray(self._reward, dtype=np.float64),
            "cost": np.asarray(self._cost, dtype=np.float64),
            "success": np.asarray(self._success, dtype=np.float64),
        }
        if self.include_state and self._qpos:
            payload["qpos"] = _stack_pad(self._qpos)
            payload["qvel"] = _stack_pad(self._qvel)
        if any(v is not None for v in self._cmd_joint_pos):
            payload["cmd_joint_pos"] = _stack_pad_optional(self._cmd_joint_pos)
        if any(v is not None for v in self._cmd_joint_torque):
            payload["cmd_joint_torque"] = _stack_pad_optional(self._cmd_joint_torque)
        if any(v is not None for v in self._intent_lin_vel):
            payload["intent_lin_vel"] = _stack_pad_optional(self._intent_lin_vel)
        np.savez(path, **payload)


def _stack_pad(arrays: List[np.ndarray]) -> np.ndarray:
    width = max(a.size for a in arrays)
    out = np.full((len(arrays), width), np.nan, dtype=np.float64)
    for i, a in enumerate(arrays):
        out[i, : a.size] = a.reshape(-1)
    return out


def _stack_pad_optional(arrays: List[Optional[np.ndarray]]) -> np.ndarray:
    nonnull = [a for a in arrays if a is not None]
    if not nonnull:
        return np.zeros((len(arrays), 0), dtype=np.float64)
    width = max(a.size for a in nonnull)
    out = np.full((len(arrays), width), np.nan, dtype=np.float64)
    for i, a in enumerate(arrays):
        if a is not None:
            out[i, : a.size] = a.reshape(-1)
    return out
