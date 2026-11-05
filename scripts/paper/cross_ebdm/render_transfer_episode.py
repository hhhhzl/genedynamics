#!/usr/bin/env python3
"""Thin paper entry point for task-owned transfer evidence visualization."""
from __future__ import annotations

import argparse
import json

from genedynamics.tasks.transfer.visualization import render_episode_evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--maximum-frames", type=int, default=120)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()
    result = render_episode_evidence(
        args.evidence,
        args.output,
        record_index=args.record_index,
        maximum_frames=args.maximum_frames,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

