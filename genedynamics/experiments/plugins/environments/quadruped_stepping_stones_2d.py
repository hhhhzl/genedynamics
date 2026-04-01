"""
Quadruped stepping-stones 2D environment plugin (continuous foothold-target).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict

import numpy as np

from genedynamics.envs.quadruped_stepping_stones_2d import (
    QuadrupedSteppingStones2DEnv,
    make_stepping_stones_energy,
)
from ...framework.base import EnvironmentPlugin


class QuadrupedSteppingStones2DPlugin(EnvironmentPlugin):
    @property
    def name(self) -> str:
        return "quadruped_stepping_stones_2d"

    def create_env(self, config: Dict[str, Any]) -> Any:
        cfg = dict(config)
        if "stance_width" in cfg:
            w = float(cfg.pop("stance_width"))
            cfg.setdefault("y_L_nominal", 0.5 * w)
            cfg.setdefault("y_R_nominal", -0.5 * w)
        if "fore_hind_offset" in cfg:
            fh = float(cfg.pop("fore_hind_offset"))
            cfg.setdefault("x_f_nominal", abs(fh))
            cfg.setdefault("x_r_nominal", -abs(fh))
        elif "l_max" in cfg:
            fh = 0.5 * float(cfg["l_max"])
            xf = min(0.32, max(0.08, fh))
            cfg.setdefault("x_f_nominal", xf)
            cfg.setdefault("x_r_nominal", -xf)
        allowed = {f.name for f in dataclasses.fields(QuadrupedSteppingStones2DEnv)}
        filtered = {k: v for k, v in cfg.items() if k in allowed}
        env = QuadrupedSteppingStones2DEnv(**filtered)
        self._centerline_y = float(getattr(env, "centerline_y", 0.0))
        return env

    def create_energy(self, env: Any = None) -> Any:
        env_obj = env if isinstance(env, QuadrupedSteppingStones2DEnv) else QuadrupedSteppingStones2DEnv()
        return make_stepping_stones_energy(env_obj)

    def get_state_dim(self) -> int:
        return 16

    def extract_position(self, state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        cy = float(getattr(self, "_centerline_y", 0.0))
        if s.size >= 16:
            return np.array([s[0], cy + s[1]], dtype=np.float32)
        if s.size >= 20:
            return np.array([s[0], cy + s[1]], dtype=np.float32)
        if s.size >= 2:
            return s[:2]
        return np.pad(s, (0, max(0, 2 - s.size)), constant_values=0.0)[:2]

    def get_position_dim(self) -> int:
        return 2
