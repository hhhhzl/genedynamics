"""
Episode writer for replay and analysis.

Saves episode data to disk in replayable format.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class EpisodeWriter:
    """
    Write episode data for replay.

    Saves:
    - states.npy, actions.npy: raw arrays
    - meta.json: config, git sha, tags
    - telemetry.json: full telemetry
    """

    def __init__(self, base_dir: str, tags: Optional[Dict[str, str]] = None):
        self.base_dir = Path(base_dir)
        self.tags = dict(tags or {})
        self._episode_count = 0

    def _git_sha(self) -> str:
        """Get current git sha."""
        try:
            out = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).resolve().parents[4],
                timeout=2,
            )
            if out.returncode == 0:
                return out.stdout.strip()[:12]
        except Exception:
            pass
        return "unknown"

    def write_episode(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        meta: Optional[Dict[str, Any]] = None,
        telemetry: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Write episode to disk.

        states: (T+1, state_dim)
        actions: (T, act_dim)
        """
        self._episode_count += 1
        ts = time.strftime("%Y%m%d-%H%M%S")
        ep_dir = self.base_dir / f"ep_{self._episode_count:04d}_{ts}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        np.save(ep_dir / "states.npy", np.asarray(states, dtype=np.float32))
        np.save(ep_dir / "actions.npy", np.asarray(actions, dtype=np.float32))

        meta_out = {
            "episode": self._episode_count,
            "timestamp": ts,
            "steps": len(actions),
            "state_dim": states.shape[-1] if states.ndim > 1 else states.size,
            "act_dim": actions.shape[-1] if actions.ndim > 1 else actions.size,
            "git_sha": self._git_sha(),
            **self.tags,
            **(meta or {}),
        }
        with open(ep_dir / "meta.json", "w") as f:
            json.dump(meta_out, f, indent=2)

        if telemetry:
            with open(ep_dir / "telemetry.json", "w") as f:
                json.dump(telemetry, f, indent=2, default=str)

        return ep_dir

    def load_episode(self, ep_dir: Path) -> Dict[str, Any]:
        """Load episode for replay."""
        states = np.load(ep_dir / "states.npy")
        actions = np.load(ep_dir / "actions.npy")
        with open(ep_dir / "meta.json") as f:
            meta = json.load(f)
        telemetry = None
        if (ep_dir / "telemetry.json").exists():
            with open(ep_dir / "telemetry.json") as f:
                telemetry = json.load(f)
        return {"states": states, "actions": actions, "meta": meta, "telemetry": telemetry}
