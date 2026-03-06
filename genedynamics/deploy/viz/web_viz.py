"""
Web visualization service for deploy pipeline.

Renders trajectory to HTML and optionally serves via Flask.
Supports: MuJoCo trajectory (GIF embed), Brax HTML (when available).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


class WebVizService:
    """
    Web visualization: save HTML and optionally serve via Flask.

    After deploy run, call render_episode() to create HTML.
    Optionally start Flask server for interactive viewing.
    """

    def __init__(
        self,
        output_dir: str,
        port: int = 5000,
        host: str = "127.0.0.1",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.port = port
        self.host = host
        self._html_path: Optional[Path] = None

    def render_episode(
        self,
        episode_dir: Union[str, Path],
        *,
        robot_type: str = "quadruped",
        model_id: str = "go2",
        format: str = "html_gif",
        width: int = 640,
        height: int = 480,
        fps: float = 20.0,
    ) -> Path:
        """
        Render episode to HTML.

        format: "html_gif" (embed GIF) | "brax_html" (Brax interactive, when env is Brax)
        """
        ep_dir = Path(episode_dir)
        if not ep_dir.exists():
            raise FileNotFoundError(f"Episode dir not found: {ep_dir}")

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        out_name = f"viz_{ep_dir.name}_{timestamp}.html"
        out_path = self.output_dir / out_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if format == "html_gif":
            self._render_mujoco_html(ep_dir, out_path, robot_type, model_id, width, height, fps)
        elif format == "brax_html":
            self._render_brax_html(ep_dir, out_path)
        else:
            self._render_mujoco_html(ep_dir, out_path, robot_type, model_id, width, height, fps)

        self._html_path = out_path
        return out_path

    def _render_mujoco_html(
        self,
        ep_dir: Path,
        out_path: Path,
        robot_type: str,
        model_id: str,
        width: int,
        height: int,
        fps: float,
    ) -> None:
        """Render MuJoCo trajectory to HTML with embedded GIF."""
        gif_path = ep_dir / "trajectory_mujoco.gif"
        if not gif_path.exists():
            self._generate_gif(ep_dir, gif_path, robot_type, model_id, width, height, fps)

        if gif_path.exists():
            with open(gif_path, "rb") as f:
                import base64
                b64 = base64.b64encode(f.read()).decode()
            html = _HTML_TEMPLATE_GIF.format(
                title=f"Deploy Viz: {ep_dir.name}",
                gif_data=f"data:image/gif;base64,{b64}",
                episode_name=ep_dir.name,
            )
        else:
            html = _HTML_TEMPLATE_NO_VIZ.format(
                title=f"Deploy Viz: {ep_dir.name}",
                message=f"Could not generate visualization for {ep_dir.name}",
            )
        with open(out_path, "w") as f:
            f.write(html)

    def _generate_gif(
        self,
        ep_dir: Path,
        gif_path: Path,
        robot_type: str,
        model_id: str,
        width: int,
        height: int,
        fps: float,
    ) -> None:
        """Generate GIF using deploy viz renderer."""
        try:
            from genedynamics.execution.logging.episode_writer import EpisodeWriter
            from genedynamics.deploy.viz.mujoco_render import render_episode_to_gif
            writer = EpisodeWriter(str(ep_dir.parent))
            data = writer.load_episode(ep_dir)
            states = data.get("states")
            if states is None:
                return
            states = np.asarray(states, dtype=np.float64)
            if states.size == 0:
                return
            render_episode_to_gif(
                ep_dir,
                states,
                actions=data.get("actions"),
                output_path=gif_path,
                model=model_id if model_id in ("go2", "ant") else "go2",
                width=width,
                height=height,
                fps=fps,
            )
        except Exception:
            pass

    def _render_brax_html(self, ep_dir: Path, out_path: Path) -> None:
        """Render Brax trajectory to interactive HTML (when Brax rollout available)."""
        try:
            import jax.numpy as jnp
            from brax import io
            states_path = ep_dir / "states.npy"
            if not states_path.exists():
                return
            states = np.load(states_path)
            if states.size == 0:
                return
            sys_path = getattr(self, "_brax_sys", None)
            if sys_path is None:
                return
            from brax.io import mjcf
            sys = mjcf.load(sys_path)
            trajectory = [jnp.array(s) for s in states]
            html = io.html.render(sys, trajectory, 1080, True)
            with open(out_path, "w") as f:
                f.write(html)
        except Exception:
            self._render_mujoco_html(ep_dir, out_path, "quadruped", "go2", 640, 480, 20.0)

    def serve(self, block: bool = True) -> None:
        """Start Flask server to serve HTML. block=True runs until Ctrl+C."""
        if self._html_path is None or not self._html_path.exists():
            raise RuntimeError("No HTML rendered. Call render_episode first.")

        try:
            import flask
        except ImportError:
            raise ImportError("Flask required for web viz. pip install flask")

        app = flask.Flask(__name__)
        html_content = self._html_path.read_text()

        @app.route("/")
        def index():
            return html_content

        print(f"Web viz: http://{self.host}:{self.port}")
        app.run(host=self.host, port=self.port, debug=False, use_reloader=False)

    def render_and_serve(
        self,
        episode_dir: Union[str, Path],
        **kwargs: Any,
    ) -> Path:
        """Render episode and start Flask server."""
        path = self.render_episode(episode_dir, **kwargs)
        self.serve(block=True)
        return path


_HTML_TEMPLATE_GIF = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: sans-serif; margin: 2em; background: #1a1a1a; color: #eee; }}
    h1 {{ font-size: 1.2em; }}
    img {{ max-width: 100%; border-radius: 8px; }}
  </style>
</head>
<body>
  <h1>Deploy Visualization: {episode_name}</h1>
  <img src="{gif_data}" alt="Trajectory" />
</body>
</html>
"""

_HTML_TEMPLATE_NO_VIZ = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: sans-serif; margin: 2em; background: #1a1a1a; color: #eee; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p>{message}</p>
</body>
</html>
"""
