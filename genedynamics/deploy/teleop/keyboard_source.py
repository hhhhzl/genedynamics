"""Keyboard teleoperation source (terminal, no GUI required).

Uses ``pynput`` to read key events on a background thread so the control loop
is never blocked. Works in a plain terminal; does **not** require a display.

Key bindings (WASD + QE + arrows)::

    W / ↑   → +vx (forward)
    S / ↓   → -vx (backward)
    A / ←   → +vy (strafe left)
    D / →   → -vy (strafe right)
    Q       → +yaw_rate (turn left)
    E       → -yaw_rate (turn right)
    R       → +body_height_delta
    F       → -body_height_delta
    Space   → stop (emergency-stop, resets on key release)
    Esc     → stop (latched until reopen)

Usage::

    src = KeyboardSource(vx_max=0.4)
    src.open()
    cmd = src.read()
    src.close()

Requires ``pynput`` (``pip install pynput``). Falls back to zero commands
if unavailable.
"""

from __future__ import annotations

import threading
from typing import Any, Optional, Set

from genedynamics.deploy.teleop.input_source import TeleopCommand

__all__ = ["KeyboardSource"]

_PYNPUT_AVAILABLE = False
try:
    from pynput import keyboard as _pynput_kbd  
    _PYNPUT_AVAILABLE = True
except ImportError:
    pass


class KeyboardSource:
    """Keyboard-driven :class:`InputSource` using pynput.

    Args:
        vx_max: Max forward velocity (m/s).
        vy_max: Max lateral velocity (m/s).
        yaw_rate_max: Max yaw rate (rad/s).
        height_delta_max: Max height delta per step (m).
    """

    name: str = "keyboard"

    def __init__(
        self,
        *,
        vx_max: float = 0.4,
        vy_max: float = 0.3,
        yaw_rate_max: float = 0.6,
        height_delta_max: float = 0.003,
    ) -> None:
        self._vx_max = float(vx_max)
        self._vy_max = float(vy_max)
        self._yr_max = float(yaw_rate_max)
        self._hd_max = float(height_delta_max)

        self._pressed: Set[str] = set()
        self._stop_latched = False
        self._lock = threading.Lock()
        self._listener: Any = None

    def open(self) -> None:
        if not _PYNPUT_AVAILABLE:
            print("[KeyboardSource] pynput not available — zero commands will be returned.")
            return
        from pynput import keyboard as kbd

        def on_press(key: Any) -> None:
            k = self._key_str(key)
            if k is not None:
                with self._lock:
                    self._pressed.add(k)
                    if k == "esc":
                        self._stop_latched = True

        def on_release(key: Any) -> None:
            k = self._key_str(key)
            if k is not None:
                with self._lock:
                    self._pressed.discard(k)

        self._listener = kbd.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()
        print("[KeyboardSource] W/S/A/D=drive  Q/E=turn  R/F=height  Space=E-stop  Esc=quit")

    def read(self) -> TeleopCommand:
        with self._lock:
            p = frozenset(self._pressed)
            stop = self._stop_latched or "space" in p

        vx = 0.0
        vy = 0.0
        yr = 0.0
        hd = 0.0

        if "w" in p or "up" in p:
            vx += self._vx_max
        if "s" in p or "down" in p:
            vx -= self._vx_max
        if "a" in p or "left" in p:
            vy += self._vy_max
        if "d" in p or "right" in p:
            vy -= self._vy_max
        if "q" in p:
            yr += self._yr_max
        if "e" in p:
            yr -= self._yr_max
        if "r" in p:
            hd += self._hd_max
        if "f" in p:
            hd -= self._hd_max

        return TeleopCommand(vx=vx, vy=vy, yaw_rate=yr, body_height_delta=hd, stop=stop)

    def close(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass

    @staticmethod
    def _key_str(key: Any) -> Optional[str]:
        try:
            from pynput.keyboard import Key
            _MAP = {
                Key.up: "up", Key.down: "down", Key.left: "left", Key.right: "right",
                Key.space: "space", Key.esc: "esc",
            }
            if key in _MAP:
                return _MAP[key]
            return key.char.lower() if hasattr(key, "char") and key.char else None
        except Exception:
            return None
