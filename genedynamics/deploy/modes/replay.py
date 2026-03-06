"""
Replay mode: load episode from disk and optionally re-render.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from genedynamics.deploy.config import DeployConfig
from genedynamics.deploy.profiles.base import RobotProfile
from genedynamics.execution.logging.episode_writer import EpisodeWriter


class ReplayMode:
    """Replay saved episodes."""

    def run(
        self,
        config: DeployConfig,
        profile: RobotProfile,
        env: Any,
        planner: Any,
    ) -> Dict[str, Any]:
        """Load and optionally render replay."""
        replay = config.replay
        if not replay or not replay.episode_dir:
            return {
                "mode": "replay",
                "error": "replay.episode_dir required",
                "results": [],
                "episode_paths": [],
                "steps": [],
            }

        base = Path(replay.episode_dir).resolve().parent
        pattern = Path(replay.episode_dir).name
        matches = list(base.glob(pattern))
        if not matches:
            ep_dir = Path(replay.episode_dir).resolve()
            if ep_dir.exists():
                matches = [ep_dir]

        if not matches:
            return {
                "mode": "replay",
                "error": f"No episodes found: {replay.episode_dir}",
                "results": [],
                "episode_paths": [],
                "steps": [],
            }

        writer = EpisodeWriter(str(base))
        results: List[Dict[str, Any]] = []
        episode_paths: List[str] = []

        for ep_dir in matches:
            data = writer.load_episode(ep_dir)
            n_states = data["states"].shape[0]
            n_actions = data["actions"].shape[0]
            results.append({
                "episode_dir": str(ep_dir),
                "states_shape": data["states"].shape,
                "actions_shape": data["actions"].shape,
                "meta": data.get("meta", {}),
            })
            episode_paths.append(str(ep_dir))
            print(f"Loaded: {ep_dir.name} - {n_states} states, {n_actions} actions")

        return {
            "mode": "replay",
            "results": results,
            "episode_paths": episode_paths,
            "steps": [r.get("states_shape", (0,))[0] - 1 for r in results],
        }
