"""AR / shared-digital-twin layer for the corridor deploy stack.

``SceneSource`` is the single authoritative obstacle definition (Vicon world
frame); ``SceneServer`` broadcasts its contract over WebSocket to AR clients
(Vision Pro / phone / tablet) and the robot side. See module docstrings.
"""

from __future__ import annotations

from genedynamics.deploy.ar.scene_source import SCHEMA_VERSION, SceneSource, Se2Transform

__all__ = ["SCHEMA_VERSION", "SceneSource", "Se2Transform", "SceneServer"]


def __getattr__(name: str):
    # Lazy so importing the package doesn't require the ``websockets`` dep
    # unless the server is actually used.
    if name == "SceneServer":
        from genedynamics.deploy.ar.scene_server import SceneServer

        return SceneServer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
