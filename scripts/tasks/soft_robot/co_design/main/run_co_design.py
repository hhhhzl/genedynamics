#!/usr/bin/env python3
"""
Run co-design experiment from YAML config.

Uses BaselineExperimentPlatform. Any registered baseline can be used.
Example: python scripts/tasks/soft_robot/co_design/main/run_co_design.py configs/soft_robot/main/crawling_ground.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Run co-design experiment")
    parser.add_argument("config", type=str, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=None, help="Override: run single seed")
    parser.add_argument("--dry-run", action="store_true", help="Validate config only")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[5]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (root / config_path).resolve()
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        return 1

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    from genedynamics.experiments.framework.baseline_platform import (
        BaselineExperimentPlatform,
        BaselineExperimentConfig,
    )
    from genedynamics.experiments.framework.baseline_registry import list_baselines
    import genedynamics.experiments.framework.baselines  # noqa: F401 — trigger registration

    if args.seed is not None:
        data["seeds"] = [args.seed]

    evaluator_params = data.get("evaluator_runtime", data.get("evaluator_params", {}))

    config = BaselineExperimentConfig(
        baseline_name=data.get("baseline_name", "mrmfmbd"),
        task_domain=data.get("task_domain", "jax_mpm"),
        task_id=data.get("task_id", "crawling_ground"),
        seeds=data.get("seeds", [0]),
        output_dir=data.get("output_dir", "results/soft_robot"),
        cache_dir=data.get("cache_dir", data.get("checkpoint_dir")),
        scheduler_config=data.get("scheduler_config"),
        method_params=data.get("method_params", {}),
        baseline_params=data.get("baseline_params", {}),
        evaluator_params=evaluator_params,
        save_gif=data.get("save_gif", False),
    )

    if config.baseline_name not in list_baselines():
        print(f"Error: Baseline '{config.baseline_name}' not found.")
        print(f"Available: {list_baselines()}")
        return 1

    if args.dry_run:
        print("Config valid (dry run)")
        print(f"  baseline: {config.baseline_name}")
        print(f"  task_id: {config.task_id}")
        print(f"  seeds: {config.seeds}")
        print(f"  output_dir: {config.output_dir}")
        return 0

    platform = BaselineExperimentPlatform(config, project_root=root)
    print(f"Running {config.baseline_name} on {config.task_id}")
    print(f"Seeds: {config.seeds}")
    results = platform.run_all()
    print(f"Completed {len(results)} runs. Results: {platform._output_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
