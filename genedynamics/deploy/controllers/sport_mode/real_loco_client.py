"""Real-hardware Unitree LocoClient wrapper.

:class:`RealLocoClient` is the production sibling of
:class:`SparkRLLocoClient`. It delegates to Unitree's stock G1 sport-mode
SDK (``unitree_sdk2py.g1.loco.g1_loco_client.LocoClient``).

Unlike the sim RL client, this class does **not** compute joint targets — the
SDK handles step planning, swing trajectories, and balance internally and
runs at its own internal control loop. The :meth:`step` method therefore:

1. Stores the latest high-level velocity / yaw-rate command.
2. Returns an **empty** joint-target dict so the
   :class:`SportModeController` knows there are no leg overrides to merge —
   the legs are commanded out-of-band by the SDK directly.

**Velocity is published from a background thread, not the control loop.** The
SDK's ``SetVelocity`` is a *blocking* RPC (it waits for the robot's loco service
to ACK). Calling it inline in the control loop means a single congested DDS
``Write`` freezes the whole loop for seconds — the robot then stops when the
command's ``duration`` lapses (the "walk 3 s / stop 5 s" symptom). So
:meth:`step` only records the latest command (non-blocking); a daemon sender
thread republishes it at ``rate_limit_hz`` with a bounded ``duration``. The
control loop runs at full rate regardless of RPC latency, and the ``duration``
acts as a dead-man (the robot halts shortly after commands stop).

This means the controller running on real G1 with :class:`RealLocoClient`
emits a :class:`ControlCommand` of kind ``"mixed"`` (loco + upper-body
joint_pos) rather than the kind ``"joint_pos"`` it produces in sim. The
:class:`UnitreeG1RobotIO` already accepts both kinds, so the only change
required to switch sim → real is one config-line change of the loco-client
class.
"""

from __future__ import annotations

import atexit
import threading
from typing import Any, Mapping, Optional, Tuple

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
        rate_limit_hz: Rate at which the background thread republishes the
            latest velocity. The command is a blocking RPC, so this caps the
            RPC traffic; 10 Hz is plenty since the bounded ``duration`` holds
            the velocity between sends.
        move_duration_s: ``SetVelocity`` duration each republish commits to. Acts
            as a dead-man: if the sender stops (loop exits / process dies), the
            robot halts after this long. Must exceed the republish period (and a
            brief RPC stall) so the gait stays continuous, but stay short for
            safety. 1.5 s.
        rpc_timeout_s: Per-call SDK RPC timeout. The stock default is ~10 s, so a
            single congested ``Write`` would block the *sender thread* that long;
            keep it short — a missed republish is harmless (``duration`` covers it).
    """

    nominal_step_period: float

    def __init__(
        self,
        *,
        sdk_client: Optional[Any] = None,
        nominal_step_period: float = 0.6,
        rate_limit_hz: float = 10.0,
        move_duration_s: float = 1.5,
        rpc_timeout_s: float = 0.5,
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
            sdk_client.SetTimeout(rpc_timeout_s)
        self._sdk = sdk_client
        self.nominal_step_period = float(nominal_step_period)
        self._period_s = 1.0 / max(rate_limit_hz, 1.0)
        self._move_duration = float(move_duration_s)

        # latest command, shared with the sender thread
        self._lock = threading.Lock()
        self._latest: Optional[Tuple[float, float, float, Optional[float]]] = None
        self._stop_evt = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._initialized = False
        atexit.register(self.stop)  # belt-and-suspenders halt on interpreter exit

    # ------------------------------------------------------------------
    # LocoClient protocol
    # ------------------------------------------------------------------

    def reset(self, pelvis_world: np.ndarray, pelvis_yaw: float = 0.0) -> None:
        """Bring the robot to a balanced stand and start the sender thread.

        ``pelvis_world`` and ``pelvis_yaw`` are accepted for protocol
        symmetry but ignored — the SDK manages its own world model.
        """
        try:
            self._sdk.BalanceStand(0)
        except Exception:  # pragma: no cover
            pass
        with self._lock:
            self._latest = None
        self._initialized = True
        self._start_worker()

    def step(
        self,
        cmd: LocoCommand,
        dt: float,
        pelvis_world: np.ndarray,
        pelvis_yaw: float,
        state: Optional[RobotState] = None,
    ) -> Mapping[str, float]:
        # Non-blocking: just record the latest command. The sender thread does the
        # (blocking) RPC, so a congested loco channel never stalls the control loop.
        del dt, pelvis_world, pelvis_yaw, state
        if not self._initialized:
            raise RuntimeError("RealLocoClient.step called before reset")
        with self._lock:
            self._latest = (
                float(cmd.vx),
                float(cmd.vy),
                float(cmd.yaw_rate),
                None if cmd.body_height is None else float(cmd.body_height),
            )
        # The SDK owns leg control; no joint overrides flow back into the
        # controller's joint vector.
        return {}

    # ------------------------------------------------------------------
    # Background velocity sender
    # ------------------------------------------------------------------

    def _start_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop_evt.clear()
        self._worker = threading.Thread(target=self._send_loop, name="loco-sender", daemon=True)
        self._worker.start()

    def _send_loop(self) -> None:
        while not self._stop_evt.is_set():
            with self._lock:
                latest = self._latest
            if latest is not None:
                vx, vy, yaw, body_height = latest
                try:
                    self._sdk.SetVelocity(vx, vy, yaw, self._move_duration)
                    if body_height is not None:
                        self._sdk.SetStandHeight(body_height)
                except Exception as exc:  # pragma: no cover
                    print(f"[RealLocoClient] WARN: SDK SetVelocity failed: {exc}")
            self._stop_evt.wait(self._period_s)  # rate-limit; interruptible

    def stop(self) -> None:
        """Halt the sender thread and command zero velocity (idempotent)."""
        self._stop_evt.set()
        worker, self._worker = self._worker, None
        if worker is not None and worker.is_alive():
            worker.join(timeout=1.0)
        try:
            self._sdk.StopMove()
        except Exception:  # pragma: no cover
            pass

    @property
    def swing_foot(self) -> Optional[str]:
        """The SDK does not expose its swing foot — return ``None``."""
        return None

    @property
    def phase(self) -> float:
        """The SDK does not expose its gait phase — return ``0``."""
        return 0.0
