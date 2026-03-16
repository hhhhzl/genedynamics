"""
Collectors for gathering report data from results directories.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from genedynamics.reports.schema import (
    DeployEpisode,
    DeploySection,
    ExperimentSection,
    LevelSummary,
    ReportData,
)


def collect_experiment_section(
    results_root: Path,
    env_name: str,
    method: str,
) -> Optional[ExperimentSection]:
    """Collect experiment data from results/{env_name}/{method}/."""
    base = results_root / env_name / method
    if not base.exists():
        return None

    overall_path = base / "overall_summary.json"
    overall_summary: Dict[str, Any] = {}
    if overall_path.exists():
        with open(overall_path) as f:
            overall_summary = json.load(f)

    level_summaries: List[LevelSummary] = []
    result_paths: List[str] = []
    total = 0

    for level_dir in sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name)):
        if not level_dir.is_dir():
            continue
        if not level_dir.name.startswith("level_"):
            continue
        try:
            level = int(level_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue

        summary_path = level_dir / "summary.json"
        summary_data: Dict[str, Any] = {}
        if summary_path.exists():
            with open(summary_path) as f:
                summary_data = json.load(f)

        metrics = {
            k.replace("avg_", ""): v
            for k, v in summary_data.items()
            if k.startswith("avg_") and isinstance(v, (int, float))
        }
        level_summaries.append(
            LevelSummary(
                level=level,
                num_experiments=summary_data.get("num_experiments", 0),
                avg_planning_time=summary_data.get("avg_planning_time", 0.0),
                std_planning_time=summary_data.get("std_planning_time", 0.0),
                metrics=metrics,
                best=summary_data.get("best"),
            )
        )
        total += summary_data.get("num_experiments", 0)

        for seed_dir in level_dir.iterdir():
            if seed_dir.is_dir() and seed_dir.name.startswith("seed_"):
                rp = seed_dir / "results.json"
                if rp.exists():
                    result_paths.append(str(rp.relative_to(base)))

    viz_paths: Dict[str, str] = {}
    viz_root = results_root / env_name / "visualizations"
    if viz_root.exists():
        for p in viz_root.rglob("*.png"):
            key = str(p.relative_to(viz_root))
            viz_paths[key] = str(p)

    return ExperimentSection(
        env_name=env_name,
        method=method,
        output_dir=str(base),
        total_experiments=total or overall_summary.get("total_experiments", 0),
        level_summaries=level_summaries,
        overall_summary=overall_summary,
        result_paths=result_paths[:100],
        viz_paths=viz_paths,
    )


def collect_deploy_section(deploy_root: Path, robot: str, mode: str) -> Optional[DeploySection]:
    """Collect deploy data from results/deploy/{robot}_{mode}/."""
    base = deploy_root / f"{robot}_{mode}"
    if not base.exists():
        return None

    episodes_dir = base / "episodes"
    episodes: List[DeployEpisode] = []
    total_steps = 0

    if episodes_dir.exists():
        for ep_dir in sorted(episodes_dir.iterdir(), key=lambda p: p.name):
            if not ep_dir.is_dir() or not ep_dir.name.startswith("ep_"):
                continue
            meta_path = ep_dir / "meta.json"
            meta: Dict[str, Any] = {}
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
            steps = meta.get("steps", 0)
            plan_times = meta.get("plan_times_ms", [])
            if not plan_times and (ep_dir / "telemetry.json").exists():
                with open(ep_dir / "telemetry.json") as f:
                    tel = json.load(f)
                    plan_times = tel.get("plan_latencies_ms", [])
            episodes.append(
                DeployEpisode(
                    episode_id=ep_dir.name,
                    steps=steps,
                    plan_times_ms=plan_times if isinstance(plan_times, list) else [],
                    path=str(ep_dir),
                )
            )
            total_steps += steps

    return DeploySection(
        robot=robot,
        mode=mode,
        output_dir=str(base),
        episodes=episodes,
        total_episodes=len(episodes),
        total_steps=total_steps,
    )


def collect_all(results_root: Path, deploy_root: Optional[Path] = None) -> ReportData:
    """Collect all report data from results directory."""
    if deploy_root is None:
        deploy_root = results_root / "deploy"

    experiment_sections: List[ExperimentSection] = []
    deploy_sections: List[DeploySection] = []

    # Experiment envs/methods
    env_methods = [
        ("single2d", "2go"),
        ("single2d", "mbd"),
        ("single2d", "cfsmbd"),
        ("single2d", "mdcoas"),
        ("double2d", "2go"),
        ("double2d", "cfsmbd"),
        ("d3il_avoiding", "dpcc"),
        ("d3il_avoiding_9d", "dpcc"),
    ]
    for env_name, method in env_methods:
        sec = collect_experiment_section(results_root, env_name, method)
        if sec is not None and (sec.total_experiments > 0 or sec.level_summaries):
            experiment_sections.append(sec)

    # Scan for any env/method not in list
    if results_root.exists():
        for env_dir in results_root.iterdir():
            if not env_dir.is_dir() or env_dir.name == "deploy":
                continue
            for method_dir in env_dir.iterdir():
                if not method_dir.is_dir():
                    continue
                key = (env_dir.name, method_dir.name)
                if key not in [(e, m) for e, m in env_methods]:
                    sec = collect_experiment_section(results_root, env_dir.name, method_dir.name)
                    if sec is not None and (sec.total_experiments > 0 or sec.level_summaries):
                        experiment_sections.append(sec)

    # Deploy
    deploy_robots = ["uav3d", "quadruped", "humanoid"]
    deploy_modes = ["sim", "shadow", "real"]
    for robot in deploy_robots:
        for mode in deploy_modes:
            sec = collect_deploy_section(deploy_root, robot, mode)
            if sec is not None and sec.total_episodes > 0:
                deploy_sections.append(sec)

    # Scan deploy dir for any robot_mode
    if deploy_root.exists():
        for d in deploy_root.iterdir():
            if not d.is_dir():
                continue
            parts = d.name.split("_", 1)
            if len(parts) == 2:
                r, m = parts[0], parts[1]
                if (r, m) not in [(s.robot, s.mode) for s in deploy_sections]:
                    sec = collect_deploy_section(deploy_root, r, m)
                    if sec is not None and sec.total_episodes > 0:
                        deploy_sections.append(sec)

    return ReportData(
        project_root=str(results_root.parent),
        experiment_sections=experiment_sections,
        deploy_sections=deploy_sections,
        meta={"results_root": str(results_root), "deploy_root": str(deploy_root)},
    )
