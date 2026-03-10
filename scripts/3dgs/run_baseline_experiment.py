#!/usr/bin/env python3
"""
Run baseline pipelines (separate from MBD method path).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Baseline config YAML")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--iters", type=int, default=30000, help="Gsplat training iterations")
    parser.add_argument("--densify", action="store_true", help="Enable gsplat densification")
    parser.add_argument("--max-gaussians", type=int, default=30000, help="Max gaussians with densify")
    parser.add_argument("--from-checkpoint", type=str, default=None, help="Warm-start checkpoint")
    parser.add_argument("--refine-lr", type=float, default=5e-4, help="LR for checkpoint refinement")
    parser.add_argument("--mask-loss", action="store_true", help="Foreground-weighted loss")
    args = parser.parse_args()

    config = Path(args.config)
    if not config.is_absolute():
        config = ROOT / config
    if not config.exists():
        print(f"Config not found: {config}")
        return 1

    cmd = [
        sys.executable,
        str(ROOT / "scripts/3dgs/train_gsplat.py"),
        str(config),
        "--iters",
        str(int(args.iters)),
    ]
    if args.output:
        cmd.extend(["--output", args.output])
    if args.densify:
        cmd.extend(["--densify", "--max-gaussians", str(int(args.max_gaussians))])
    if args.mask_loss:
        cmd.append("--mask-loss")
    if args.from_checkpoint:
        ckpt = Path(args.from_checkpoint)
        if not ckpt.is_absolute():
            ckpt = ROOT / ckpt
        cmd.extend(["--from-checkpoint", str(ckpt), "--refine-lr", str(float(args.refine_lr))])

    print("Running baseline training:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
