"""
Web visualization service for deploy pipeline.

Renders trajectory to HTML and optionally serves via Flask.
Supports: MuJoCo trajectory (GIF embed), Brax HTML (when available).
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from genedynamics.viz.motion_episode import MotionEpisode
from genedynamics.viz.motion_renderer import MotionRenderer


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
        self._motion_renderer = MotionRenderer(self.output_dir)

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

        use_brax = format == "brax_html"
        if format == "auto" and (ep_dir / "meta.json").exists():
            import json
            try:
                with open(ep_dir / "meta.json") as f:
                    meta = json.load(f)
                use_brax = "brax" in str(meta.get("env_name", "")).lower()
            except Exception:
                pass
        if format == "html_gif":
            self._render_mujoco_html(ep_dir, out_path, robot_type, model_id, width, height, fps)
        elif use_brax:
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
            gif_src = Path(
                os.path.relpath(gif_path.resolve(), out_path.parent.resolve())
            ).as_posix()
            try:
                if gif_path.stat().st_size <= 1_500_000:
                    with open(gif_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode()
                    gif_src = f"data:image/gif;base64,{b64}"
            except OSError:
                pass
            html = _HTML_TEMPLATE_GIF.format(
                title=f"Deploy Viz: {ep_dir.name}",
                gif_src=gif_src,
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
            from genedynamics.deploy.viz.mujoco_render import render_episode_to_gif

            episode = MotionEpisode.from_deploy_episode_dir(ep_dir, fps=fps)
            if episode.states.size == 0:
                return
            render_episode_to_gif(
                ep_dir,
                episode.states,
                actions=episode.actions,
                output_path=gif_path,
                model=model_id if model_id in ("go2", "ant") else "go2",
                width=width,
                height=height,
                fps=float(episode.fps),
            )
        except Exception:
            pass

    def render_motion_episode(
        self,
        episode: MotionEpisode,
        *,
        name: str = "motion",
        width: int = 640,
        height: int = 480,
        fps: Optional[float] = None,
    ) -> Path:
        """
        Render an in-memory motion episode to HTML.

        This is a shared entry-point that can be used by non-deploy workflows
        (e.g. experiments) without writing deploy episode artifacts first.
        """
        out_path = self._motion_renderer.render_html(
            episode,
            name=name,
            width=width,
            height=height,
            fps=fps,
        )
        self._html_path = out_path
        return out_path

    def _render_brax_html(self, ep_dir: Path, out_path: Path) -> None:
        """Render Brax trajectory to interactive HTML. Converts flat states to pipeline_states."""
        try:
            from brax import io
            states_path = ep_dir / "states.npy"
            meta_path = ep_dir / "meta.json"
            if not states_path.exists():
                self._render_mujoco_html(ep_dir, out_path, "quadruped", "go2", 640, 480, 20.0)
                return
            states = np.load(states_path)
            if states.ndim == 1:
                states = states.reshape(1, -1)
            if states.size == 0:
                self._render_mujoco_html(ep_dir, out_path, "quadruped", "go2", 640, 480, 20.0)
                return
            state_dim = int(states.shape[-1])
            env_name = None
            if meta_path.exists():
                import json
                with open(meta_path) as f:
                    meta = json.load(f)
                env_name = meta.get("env_name")
            if not env_name:
                env_name = "quadruped_go2_brax" if state_dim >= 35 else "humanoid_run_brax"
            env = self._load_brax_env_for_render(env_name)
            if env is None:
                self._render_mujoco_html(ep_dir, out_path, "quadruped", "go2", 640, 480, 20.0)
                return
            sys = env._env.sys
            trajectory = [env._env.pipeline_init(s[: env._nq], s[env._nq :]) for s in states]
            html_str = io.html.render(sys, trajectory)
            with open(out_path, "w") as f:
                f.write(html_str)
        except Exception:
            self._render_mujoco_html(ep_dir, out_path, "quadruped", "go2", 640, 480, 20.0)

    def _load_brax_env_for_render(self, env_name: str):
        """Load Brax env for HTML render. Returns BraxFlatEnv or None."""
        try:
            if "go2" in env_name.lower():
                from genedynamics.envs.brax_env import make_brax_go2
                return make_brax_go2()
            if "humanoid" in env_name.lower():
                from genedynamics.envs.brax_env import make_brax_humanoid_run
                return make_brax_humanoid_run()
        except Exception:
            pass
        return None

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
  <img src="{gif_src}" alt="Trajectory" />
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
