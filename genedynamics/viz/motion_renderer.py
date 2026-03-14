"""
Shared renderer for robot motion episodes (GIF + HTML wrapper).
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Optional

from genedynamics.viz.motion_episode import MotionEpisode


class MotionRenderer:
    """
    Render a MotionEpisode to assets that are easy to share and inspect.
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def render_gif(
        self,
        episode: MotionEpisode,
        *,
        name: str = "motion",
        width: int = 640,
        height: int = 480,
        fps: Optional[float] = None,
        draw_trajectory: bool = True,
    ) -> Path:
        ts = time.strftime("%Y%m%d-%H%M%S")
        gif_path = self.output_dir / f"{name}_{ts}.gif"
        # Lazy import avoids package-level circular import with deploy.viz.web_viz.
        from genedynamics.deploy.viz.mujoco_render import render_episode_to_gif

        return render_episode_to_gif(
            episode_dir=self.output_dir,
            states=episode.states,
            actions=episode.actions,
            output_path=gif_path,
            model=episode.model_id,
            width=width,
            height=height,
            fps=float(fps if fps is not None else episode.fps),
            draw_trajectory=draw_trajectory,
        )

    def render_html(
        self,
        episode: MotionEpisode,
        *,
        name: str = "motion",
        width: int = 640,
        height: int = 480,
        fps: Optional[float] = None,
        draw_trajectory: bool = True,
    ) -> Path:
        gif_path = self.render_gif(
            episode,
            name=name,
            width=width,
            height=height,
            fps=fps,
            draw_trajectory=draw_trajectory,
        )
        html_path = gif_path.with_suffix(".html")
        with open(gif_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        html = _HTML_TEMPLATE.format(
            title=f"Motion Replay: {name}",
            subtitle=f"robot={episode.robot_type}, model={episode.model_id}, frames={len(episode.states)}",
            gif_data=f"data:image/gif;base64,{b64}",
        )
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        return html_path


_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: sans-serif; margin: 2em; background: #111; color: #eee; }}
    h1 {{ font-size: 1.2em; margin-bottom: 0.2em; }}
    p {{ opacity: 0.85; margin-top: 0; }}
    img {{ max-width: 100%; border-radius: 8px; border: 1px solid #333; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p>{subtitle}</p>
  <img src="{gif_data}" alt="Motion replay GIF" />
</body>
</html>
"""
