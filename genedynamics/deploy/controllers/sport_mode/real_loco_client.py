"""Real-hardware Unitree LocoClient wrapper.

:class:`RealLocoClient` is the production sibling of
:class:`SparkRLLocoClient`. It delegates to Unitree's stock G1 sport-mode
SDK (``unitree_sdk2py.g1.loco.g1_loco_client.LocoClient``).

Unlike the sim RL client, this class does **not** compute joint targets — the
SDK handles step planning, swing trajectories, and balance internally and
runs at its own internal control loop. The :meth:`step` method therefore:

1. Issues the high-level velocity / yaw-rate command via ``Move``.
2. Returns an **empty** joint-target dict so the
   :class:`SportModeController` knows there are no leg overrides to merge —
   the legs are commanded out-of-band by the SDK directly.

This means the controller running on real G1 with :class:`RealLocoClient`
emits a :class:`ControlCommand` of kind ``"mixed"`` (loco + upper-body
joint_pos) rather than the kind ``"joint_pos"`` it produces in sim. The
:class:`UnitreeG1RobotIO` already accepts both kinds, so the only change
required to switch sim → real is one config-line change of the loco-client
class.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from genedynamics.deploy.interfaces.messages import LocoCommand, RobotState

__all__ = ["RealLocoClient"]


class RealLocoClient:
    """Wraps the Unitree SDK ``LocoClient`` as a :class:`LocoClient`.

    Args:
        sdk_client: Optional pre-initialized
            ``unitree_sdk2py.g1.loco.g1_loco_client.LocoClient``. When
            ``None``, the SDK is imported and a new client is constructed
            (this requires a working network channel — typically already
            initialized by :class:`UnitreeG1RobotIO`).
        nominal_step_period: Step period in seconds. Used purely as
            telemetry; the SDK has its own gait clock and does not expose it.
        rate_limit_hz: Maximum frequency at which ``Move`` is republished.
            The SDK accepts updates at any rate but spamming wastes
            bandwidth; 50 Hz is a sensible default for high-level intent.
    """

    nominal_step_period: float

    def __init__(
        self,
        *,
        sdk_client: Optional[Any] = None,
        nominal_step_period: float = 0.6,
        rate_limit_hz: float = 50.0,
    ) -> None:
        if sdk_client is None:
            try:
                from unitree_sdk2py.g1.loco.g1_loco_client import (
                    LocoClient as SdkLocoClient,
                )
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "RealLocoClient requires unitree_sdk2py. Use SparkRLLocoClient "
                    "for sim-only runs."
                ) from exc
            sdk_client = SdkLocoClient()
            sdk_client.Init()
            sdk_client.SetTimeout(10.0)
        self._sdk = sdk_client
        self.nominal_step_period = float(nominal_step_period)
        self._min_period_s = 1.0 / max(rate_limit_hz, 1.0)
        self._t_since_last_send = 0.0
        self._last_cmd: Optional[LocoCommand] = None
        self._initialized = False

    # ------------------------------------------------------------------
    # LocoClient protocol
    # ------------------------------------------------------------------

    def reset(self, pelvis_world: np.ndarray, pelvis_yaw: float = 0.0) -> None:
        """Bring the robot to a balanced stand and clear local state.

        ``pelvis_world`` and ``pelvis_yaw`` are accepted for protocol
        symmetry but ignored — the SDK manages its own world model.
        """
        try:
            self._sdk.BalanceStand(0)
        except Exception:  # pragma: no cover
            pass
        self._t_since_last_send = float("inf")
        self._last_cmd = None
        self._initialized = True

    def step(
        self,
        cmd: LocoCommand,
        dt: float,
        pelvis_world: np.ndarray,
        pelvis_yaw: float,
        state: Optional[RobotState] = None,
    ) -> Mapping[str, float]:
        # state is accepted for protocol compatibility — the SDK manages its
        # own world model and doesn't need our local snapshot.
        del state
        if not self._initialized:
            raise RuntimeError("RealLocoClient.step called before reset")
        self._t_since_last_send += float(dt)
        if (
            self._t_since_last_send >= self._min_period_s
            or self._last_cmd is None
            or _cmd_changed(cmd, self._last_cmd)
        ):
            try:
                self._sdk.Move(float(cmd.vx), float(cmd.vy), float(cmd.yaw_rate))
                if cmd.body_height is not None:
                    self._sdk.SetStandHeight(float(cmd.body_height))
            except Exception as exc:  # pragma: no cover
                print(f"[RealLocoClient] WARN: SDK Move failed: {exc}")
            self._t_since_last_send = 0.0
            self._last_cmd = cmd
        # The SDK owns leg control; no joint overrides flow back into the
        # controller's joint vector.
        return {}

    @property
    def swing_foot(self) -> Optional[str]:
        """The SDK does not expose its swing foot — return ``None``."""
        return None

    @property
    def phase(self) -> float:
        """The SDK does not expose its gait phase — return ``0``."""
        return 0.0


def _cmd_changed(a: LocoCommand, b: LocoCommand, eps: float = 1e-3) -> bool:
    if abs(a.vx - b.vx) > eps or abs(a.vy - b.vy) > eps:
        return True
    if abs(a.yaw_rate - b.yaw_rate) > eps:
        return True
    if (a.body_height is None) != (b.body_height is None):
        return True
    if a.body_height is not None and b.body_height is not None:
        if abs(a.body_height - b.body_height) > eps:
            return True
    return False
