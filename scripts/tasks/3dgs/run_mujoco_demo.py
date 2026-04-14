#!/usr/bin/env python3
"""
Exp3: MuJoCo closed-loop active-perception demo.

This script thin-wraps `run_active_selection.py` for the MuJoCo env:
it loads the scene XML, renders candidate views, runs the active loop.

Usage:
  python scripts/tasks/3dgs/run_mujoco_demo.py \
      configs/3dgs/exp3_mujoco/mbd/active_bias.yaml \
      --output results/3dgs/exp3_mujoco/mbd_active_bias
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp3 MuJoCo demo")
    parser.add_argument("config", type=str)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--scorer", type=str, default="posterior_variance")
    parser.add_argument("--n-init", type=int, default=5)
    parser.add_argument("--n-rounds", type=int, default=10)
    parser.add_argument("--n-select", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(ROOT / "scripts/tasks/3dgs/run_active_selection.py"),
        args.config,
        "--scorer", args.scorer,
        "--n-init", str(args.n_init),
        "--n-rounds", str(args.n_rounds),
        "--n-select", str(args.n_select),
        "--output", args.output,
        "--seed", str(args.seed),
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
