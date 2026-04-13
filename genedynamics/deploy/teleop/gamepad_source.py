"""Gamepad / joystick input source (pygame backend).

Maps a standard gamepad (Xbox / DualShock layout) to :class:`TeleopCommand`.

Button / axis mapping (Xbox layout, configurable via ``axis_map`` /
``button_map``)::

    Left stick  X  → vy  (lateral)
    Left stick  Y  → vx  (forward, negated — stick up = forward)
    Right stick X  → yaw_rate
    Right stick Y  → body_height_delta
    LB / RB        → arm presets (extras["arm_preset"] = "left"/"right")
    Start          → stop (emergency-stop)

Usage::

    src = GamepadSource(vx_max=0.5, vy_max=0.3, yaw_rate_max=0.8)
    src.open()
    cmd = src.read()     # call at control frequency
    src.close()

Requires ``pygame`` (``pip install pygame``). If pygame is unavailable or no
gamepad is detected the source silently returns zero commands (``stop=False``)
so tests can run headlessly.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

from genedynamics.deploy.teleop.input_source import TeleopCommand

__all__ = ["GamepadSource"]

# Axis indices (pygame default for most XInput-compatible gamepads)
_AXIS_LX = 0   # left stick X  → vy
_AXIS_LY = 1   # left stick Y  → vx (negated)
_AXIS_RX = 2   # right stick X → yaw_rate
_AXIS_RY = 3   # right stick Y → body_height_delta
_BTN_START = 7
_BTN_LB = 4
_BTN_RB = 5
_DEADZONE = 0.08


class GamepadSource:
    """Pygame-backed gamepad input source.

    Args:
        joystick_index: pygame joystick index (0 = first detected gamepad).
        vx_max: Maximum forward velocity (m/s).
        vy_max: Maximum lateral velocity (m/s).
        yaw_rate_max: Maximum yaw rate (rad/s).
        height_delta_max: Maximum height delta per step (m).
        axis_map: Override default axis indices as ``{"vx": N, "vy": N, ...}``.
        button_map: Override default button indices as ``{"stop": N, ...}``.
        deadzone: Axis deadzone radius (0–1). Values below this are zeroed.
    """

    name: str = "gamepad"

    def __init__(
        self,
        *,
        joystick_index: int = 0,
        vx_max: float = 0.5,
        vy_max: float = 0.3,
        yaw_rate_max: float = 0.8,
        height_delta_max: float = 0.005,
        axis_map: Optional[Dict[str, int]] = None,
        button_map: Optional[Dict[str, int]] = None,
        deadzone: float = _DEADZONE,
    ) -> None:
        self._joy_idx = int(joystick_index)
        self._vx_max = float(vx_max)
        self._vy_max = float(vy_max)
        self._yr_max = float(yaw_rate_max)
        self._hd_max = float(height_delta_max)
        self._deadzone = float(deadzone)

        am = axis_map or {}
        self._ax_lx = int(am.get("vy", _AXIS_LX))
        self._ax_ly = int(am.get("vx", _AXIS_LY))
        self._ax_rx = int(am.get("yaw_rate", _AXIS_RX))
        self._ax_ry = int(am.get("height", _AXIS_RY))

        bm = button_map or {}
        self._btn_stop = int(bm.get("stop", _BTN_START))
        self._btn_lb = int(bm.get("lb", _BTN_LB))
        self._btn_rb = int(bm.get("rb", _BTN_RB))

        self._joystick: Any = None
        self._pygame: Any = None
        self._lock = threading.Lock()
        self._latest = TeleopCommand()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._available = False

    def open(self) -> None:
        """Initialise pygame and start the polling thread."""
        try:
            import pygame
            self._pygame = pygame
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() == 0:
                print(f"[GamepadSource] No joystick detected — zero commands will be returned.")
                return
            self._joystick = pygame.joystick.Joystick(self._joy_idx)
            self._joystick.init()
            self._available = True
            print(f"[GamepadSource] Using joystick: {self._joystick.get_name()}")
        except ImportError:
            print("[GamepadSource] pygame not available — zero commands will be returned.")
            return

        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="gamepad-poll")
        self._thread.start()

    def read(self) -> TeleopCommand:
        """Return the latest polled command (non-blocking)."""
        with self._lock:
            return TeleopCommand(
                vx=self._latest.vx,
                vy=self._latest.vy,
                yaw_rate=self._latest.yaw_rate,
                body_height_delta=self._latest.body_height_delta,
                arm_delta=dict(self._latest.arm_delta),
                stop=self._latest.stop,
                extras=dict(self._latest.extras),
            )

    def close(self) -> None:
        """Stop the polling thread and release pygame."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._pygame is not None:
            try:
                self._pygame.joystick.quit()
                self._pygame.quit()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        """Background thread — polls pygame events at ~200 Hz."""
        import time
        pygame = self._pygame
        joy = self._joystick
        while self._running:
            for _ in pygame.event.get():
                pass  # drain event queue so pygame updates axis values

            vx = self._apply_deadzone(-joy.get_axis(self._ax_ly)) * self._vx_max
            vy = self._apply_deadzone(-joy.get_axis(self._ax_lx)) * self._vy_max
            yr = self._apply_deadzone(-joy.get_axis(self._ax_rx)) * self._yr_max
            hd = self._apply_deadzone(-joy.get_axis(self._ax_ry)) * self._hd_max

            try:
                stop = bool(joy.get_button(self._btn_stop))
                lb = bool(joy.get_button(self._btn_lb))
                rb = bool(joy.get_button(self._btn_rb))
            except Exception:
                stop, lb, rb = False, False, False

            extras: Dict[str, Any] = {}
            if lb:
                extras["arm_preset"] = "left"
            elif rb:
                extras["arm_preset"] = "right"

            cmd = TeleopCommand(
                vx=float(vx),
                vy=float(vy),
                yaw_rate=float(yr),
                body_height_delta=float(hd),
                stop=stop,
                extras=extras,
            )
            with self._lock:
                self._latest = cmd

            time.sleep(0.005)  # ~200 Hz

    def _apply_deadzone(self, value: float) -> float:
        v = float(value)
        if abs(v) < self._deadzone:
            return 0.0
        sign = 1.0 if v > 0 else -1.0
        return sign * (abs(v) - self._deadzone) / (1.0 - self._deadzone)
