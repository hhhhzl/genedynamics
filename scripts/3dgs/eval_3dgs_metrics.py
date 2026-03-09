#!/usr/bin/env python3
"""
Evaluate 3DGS experiment metrics: PSNR, LPIPS, NLL, inference budget.

Usage:
  python scripts/3dgs/eval_3dgs_metrics.py results/3dgs/lego_mbd_iid
  python scripts/3dgs/eval_3dgs_metrics.py results/3dgs/lego_3dgs_map --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def compute_psnr(pred: np.ndarray, gt: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Compute PSNR (dB)."""
    mse = np.mean((np.asarray(pred) - np.asarray(gt)) ** 2)
    if mask is not None:
        mse = np.sum((pred - gt) ** 2 * mask) / (np.sum(mask) + 1e-8)
    if mse < 1e-10:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


def compute_lpips(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute LPIPS if lpips available, else return -1."""
    try:
        import lpips
        loss_fn = lpips.LPIPS(net="alex")
        pred_t = np.clip(pred, 0, 1).astype(np.float32)
        gt_t = np.clip(gt, 0, 1).astype(np.float32)
        if pred_t.ndim == 3:
            pred_t = pred_t[None, ...]
            gt_t = gt_t[None, ...]
        # lpips expects (N,C,H,W), values in [-1,1] or [0,1]
        p = (pred_t * 2 - 1).transpose(0, 3, 1, 2)
        g = (gt_t * 2 - 1).transpose(0, 3, 1, 2)
        d = loss_fn(p, g)
        return float(np.mean(d))
    except ImportError:
        return -1.0


def load_results(result_dir: Path) -> Dict[str, Any]:
    """Load results.json from experiment output."""
    candidates = [
        result_dir / "level_0" / "seed_0" / "results.json",
        result_dir / "results.json",
        result_dir / "summary.json",
    ]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Evaluate 3DGS metrics")
    parser.add_argument("result_dir", type=str, help="Path to results (e.g. results/3dgs/lego_mbd_iid)")
    parser.add_argument("--split", type=str, default="test", help="train|val|test")
    parser.add_argument("--output", type=str, default=None, help="Output metrics JSON path")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        print(f"Result dir not found: {result_dir}")
        return 1

    data = load_results(result_dir)
    metrics: Dict[str, Any] = {
        "result_dir": str(result_dir),
        "split": args.split,
        "psnr": None,
        "lpips": None,
        "nll": None,
        "inference_budget": None,
    }

    if "total_log_prob" in data:
        metrics["nll"] = -float(data["total_log_prob"])
    if "metrics" in data and isinstance(data["metrics"], dict):
        m = data["metrics"]
        if "psnr" in m:
            metrics["psnr"] = float(m["psnr"])
        if "lpips" in m:
            metrics["lpips"] = float(m["lpips"])
        if "nll" in m:
            metrics["nll"] = float(m["nll"])
    if "scene_params" in data or "candidate_costs" in data:
        costs = data.get("candidate_costs", [])
        if costs:
            metrics["best_cost"] = float(np.min(costs))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {out_path}")

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    exit(main())
