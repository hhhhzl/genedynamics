"""
Stepping-stones 2D scene generator plugin.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from genedynamics.envs.obstacles.base import ObstacleManager
from genedynamics.envs.obstacles.stepping_stones import make_stepping_stones_obstacles
from genedynamics.tasks.stepping_stones import sample_stepping_stones_scene
from ...framework.base import ObstacleGeneratorPlugin


class SteppingStones2DObstacleGeneratorPlugin(ObstacleGeneratorPlugin):
    @property
    def name(self) -> str:
        return "stepping_stones_2d"

    def generate(
        self,
        level: int,
        seed: int,
        start_pos: np.ndarray,
        target_pos: np.ndarray,
        config: Dict[str, Any],
    ) -> ObstacleManager:
        s = np.asarray(start_pos, dtype=np.float32).reshape(-1)
        t = np.asarray(target_pos, dtype=np.float32).reshape(-1)
        if "start_mid" in config:
            start_mid = np.asarray(config["start_mid"], dtype=np.float32).reshape(-1)[:2].tolist()
        elif s.size >= 16 and str(config.get("env_name", "")) == "quadruped_stepping_stones_2d":
            start_mid = [float(s[0]), float(s[1])]
        elif s.size >= 4:
            start_mid = (0.5 * (s[:2] + s[2:4])).tolist()
        else:
            start_mid = (s[:2] if s.size >= 2 else np.array([-1.25, 0.0], dtype=np.float32)).tolist()
        if "goal_mid" in config:
            goal_mid = np.asarray(config["goal_mid"], dtype=np.float32).reshape(-1)[:2].tolist()
        elif t.size >= 4:
            goal_mid = (0.5 * (t[:2] + t[2:4])).tolist()
        else:
            goal_mid = (t[:2] if t.size >= 2 else np.array([1.25, 0.0], dtype=np.float32)).tolist()

        lmx = float(config.get("l_max", 0.35))
        max_lane = config.get("max_lane_center_dx", None)
        cap = config.get("k_horizon_cap", None)
        fore_hind_offset = float(config.get("fore_hind_offset", 0.5 * lmx))
        scene = sample_stepping_stones_scene(
            level=max(1, int(level)),
            seed=int(seed),
            l_max=lmx,
            stance_width=float(config.get("stance_width", 0.30)),
            start_mid=(float(start_mid[0]), float(start_mid[1])),
            goal_mid=(float(goal_mid[0]), float(goal_mid[1])),
            fore_hind_offset=fore_hind_offset,
            max_lane_center_dx=(float(max_lane) if max_lane is not None else None),
            lane_gap_l_ref=float(config.get("lane_gap_l_ref", 0.35)),
            lane_gap_scale=float(config.get("lane_gap_scale", 1.0)),
            lane_tail_margin=int(config.get("lane_tail_margin", 4)),
            k_horizon_cap=(int(cap) if cap is not None else None),
        )
        return make_stepping_stones_obstacles(scene)
