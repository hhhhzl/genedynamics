"""
Report data schema for unified experiment and deploy results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class LevelSummary:
    """Per-level experiment summary."""

    level: int
    num_experiments: int
    avg_planning_time: float
    std_planning_time: float
    metrics: Dict[str, float]
    best: Optional[Dict[str, float]] = None


@dataclass
class ExperimentSection:
    """Experiment run section (single2d, double2d, etc.)."""

    env_name: str
    method: str
    output_dir: str
    total_experiments: int
    level_summaries: List[LevelSummary]
    overall_summary: Dict[str, Any]
    result_paths: List[str] = field(default_factory=list)
    viz_paths: Dict[str, str] = field(default_factory=dict)


@dataclass
class DeployEpisode:
    """Deploy episode record."""

    episode_id: str
    steps: int
    plan_times_ms: List[float]
    path: str


@dataclass
class DeploySection:
    """Deploy run section (uav3d_sim, quadruped_sim, humanoid_sim)."""

    robot: str
    mode: str  # sim, shadow, real
    output_dir: str
    episodes: List[DeployEpisode]
    total_episodes: int
    total_steps: int


@dataclass
class ReportData:
    """Full report data structure."""

    title: str = "Enerdynamics Experiment Report"
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    project_root: str = ""
    experiment_sections: List[ExperimentSection] = field(default_factory=list)
    deploy_sections: List[DeploySection] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
