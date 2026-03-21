#!/usr/bin/env python3
"""
Experiment 1 stress sweeps (Part B/C): merge env_params into a base YAML and run MBD.

Each run writes results under output_root/<run_name>/ (config_resolved.yaml + run_full_experiment output).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def _deep_merge_dict(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="YAML-driven MBD3D stress sweep")
    parser.add_argument("sweep_yaml", type=str, help="Sweep spec (base_config, runs, output_root)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing",
    )
    parser.add_argument(
        "--extra-args",
        nargs="*",
        default=[],
        help="Extra args passed to run_full_experiment.py (e.g. --n-seeds 4)",
    )
    args = parser.parse_args()

    try:
        import yaml
    except ImportError as e:
        print("PyYAML is required: pip install pyyaml", file=sys.stderr)
        return 1

    sweep_path = Path(args.sweep_yaml)
    if not sweep_path.is_absolute():
        sweep_path = ROOT / sweep_path
    if not sweep_path.exists():
        print(f"Sweep file not found: {sweep_path}", file=sys.stderr)
        return 1

    with open(sweep_path, encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    base_rel = spec.get("base_config")
    if not base_rel:
        print("sweep_yaml must contain 'base_config'", file=sys.stderr)
        return 1
    base_path = Path(base_rel)
    if not base_path.is_absolute():
        base_path = ROOT / base_path
    if not base_path.exists():
        print(f"base_config not found: {base_path}", file=sys.stderr)
        return 1

    with open(base_path, encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    out_root = Path(spec.get("output_root", "results/3dgs/stress_runs"))
    if not out_root.is_absolute():
        out_root = ROOT / out_root

    runs = spec.get("runs") or []
    if not runs:
        print("No runs specified", file=sys.stderr)
        return 1

    summary: list[dict] = []
    runner = ROOT / "scripts" / "tasks" / "3dgs" / "run_full_experiment.py"

    for run in runs:
        name = run.get("name")
        if not name:
            print("Each run must have 'name'", file=sys.stderr)
            return 1
        cfg = deepcopy(base_cfg)
        cfg["env_params"] = _deep_merge_dict(cfg.get("env_params") or {}, run.get("env_params") or {})
        if run.get("method_params"):
            cfg["method_params"] = _deep_merge_dict(
                cfg.get("method_params") or {},
                run["method_params"],
            )
        run_dir = out_root / str(name)
        run_dir.mkdir(parents=True, exist_ok=True)
        resolved = run_dir / "config_resolved.yaml"
        cfg["output_dir"] = str(run_dir)
        with open(resolved, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

        cmd = [sys.executable, str(runner), str(resolved), *args.extra_args]
        print("Running:", " ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True, cwd=str(ROOT))
            metrics_path = run_dir / "metrics.json"
            row: dict = {"name": str(name), "output_dir": str(run_dir)}
            if metrics_path.exists():
                with open(metrics_path, encoding="utf-8") as mf:
                    row["metrics"] = json.load(mf)
            summary.append(row)

    if not args.dry_run and summary:
        with open(out_root / "sweep_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {out_root / 'sweep_summary.json'}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
