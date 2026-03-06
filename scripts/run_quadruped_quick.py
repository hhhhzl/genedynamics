#!/usr/bin/env python3
"""
One-shot: run quadruped MBD deploy + render GIF.
Uses mbd_deploy_quick.yaml (Go2).
"""
import subprocess
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]

def main():
    cfg = "configs/quadruped/flat/mbd_deploy_quick.yaml"
    out = "results/deploy/quadruped_go2_mbd_sim"
    # 1. Deploy
    r = subprocess.run([
        sys.executable, "-m", "genedynamics.deploy.cli",
        "--config", cfg,
        "--episodes", "2",
        "--max-steps", "60",
    ], cwd=str(_root))
    if r.returncode != 0:
        return r.returncode
    # 2. Render
    r = subprocess.run([
        sys.executable, str(_root / "scripts" / "render_deploy_quadruped_gif.py"),
        out,
        "--episodes", "2",
        "--model", "auto",
    ], cwd=str(_root))
    return r.returncode

if __name__ == "__main__":
    sys.exit(main())
