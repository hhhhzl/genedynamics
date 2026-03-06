"""
Sim2Sim launcher: start sim process, then plan process after delay.

Usage:
    genedynamics-deploy-sim2sim --config configs/quadruped/flat/mbd_deploy.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parents[3]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch sim + plan (sim2sim)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds before starting plan")
    parser.add_argument("--sim-only", action="store_true", help="Only run sim process")
    parser.add_argument("--plan-only", action="store_true", help="Only run plan process")
    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"Config not found: {args.config}")
        return 1

    sim_cmd = [
        sys.executable,
        "-m", "genedynamics.deploy.sim_plan.sim_process",
        "--config", args.config,
    ]
    plan_cmd = [
        sys.executable,
        "-m", "genedynamics.deploy.sim_plan.plan_process",
        "--config", args.config,
    ]

    if args.plan_only:
        return subprocess.run(plan_cmd).returncode

    sim_proc = subprocess.Popen(sim_cmd)
    if args.sim_only:
        sim_proc.wait()
        return sim_proc.returncode or 0

    time.sleep(args.delay)
    plan_proc = subprocess.Popen(plan_cmd)
    try:
        sim_proc.wait()
        plan_proc.wait()
    except KeyboardInterrupt:
        sim_proc.terminate()
        plan_proc.terminate()
        sim_proc.wait()
        plan_proc.wait()
    return 0


if __name__ == "__main__":
    sys.exit(main())
