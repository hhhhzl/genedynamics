#!/usr/bin/env python3
"""
Plot co-design experiment results for paper figures.

Reads results.json files from phase output directories and generates:
  1. Return vs wall-clock (Exp-1: S3 multi-fidelity efficiency)
  2. ID vs OOD bar chart (Exp-2: S1 mode robustness)
  3. Mode responsibilities heatmap
  4. Fidelity usage histogram

Usage:
    python plot_co_design_results.py --results-root results/ --output-dir figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


def load_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load results.json from a phase directory."""
    path = results_dir / "results.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def extract_returns_and_walltime(
    results: List[Dict[str, Any]],
) -> Tuple[List[float], List[float]]:
    """Extract (returns, wall_times) from results list."""
    returns, wall_times = [], []
    for r in results:
        res = r.get("result", {})
        returns.append(float(res.get("return_", 0.0)))
        wall_times.append(float(r.get("wall_time", 0.0)))
    return returns, wall_times


# ── Exp-1: S3 Multi-Fidelity Efficiency ──────────────────────────────────────


def plot_return_vs_wallclock(
    results_by_method: Dict[str, List[Dict[str, Any]]],
    output_path: Path,
) -> None:
    """Bar chart: final return and wall-clock for single-fidelity vs multi-fidelity."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    methods = list(results_by_method.keys())
    returns_mean, returns_std = [], []
    walltime_mean, walltime_std = [], []

    for method in methods:
        rets, wts = extract_returns_and_walltime(results_by_method[method])
        returns_mean.append(np.mean(rets) if rets else 0)
        returns_std.append(np.std(rets) if len(rets) > 1 else 0)
        walltime_mean.append(np.mean(wts) if wts else 0)
        walltime_std.append(np.std(wts) if len(wts) > 1 else 0)

    x = np.arange(len(methods))
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974"]

    ax = axes[0]
    ax.bar(x, returns_mean, yerr=returns_std, color=colors[: len(methods)], capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("Final Return")
    ax.set_title("Return (higher is better)")

    ax = axes[1]
    ax.bar(x, walltime_mean, yerr=walltime_std, color=colors[: len(methods)], capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("Wall-clock (s)")
    ax.set_title("Wall-clock Time (lower is better)")

    fig.suptitle("Exp-1: Multi-Fidelity Efficiency (S3)")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── Exp-2: S1 Mode Robustness ────────────────────────────────────────────────


def plot_id_vs_ood(
    id_results: Dict[str, List[Dict[str, Any]]],
    ood_results: Dict[str, List[Dict[str, Any]]],
    output_path: Path,
) -> None:
    """Grouped bar chart: ID vs OOD return for each method."""
    methods = sorted(set(id_results.keys()) | set(ood_results.keys()))
    id_means, id_stds = [], []
    ood_means, ood_stds = [], []

    for m in methods:
        rets_id, _ = extract_returns_and_walltime(id_results.get(m, []))
        rets_ood, _ = extract_returns_and_walltime(ood_results.get(m, []))
        id_means.append(np.mean(rets_id) if rets_id else 0)
        id_stds.append(np.std(rets_id) if len(rets_id) > 1 else 0)
        ood_means.append(np.mean(rets_ood) if rets_ood else 0)
        ood_stds.append(np.std(rets_ood) if len(rets_ood) > 1 else 0)

    x = np.arange(len(methods))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, id_means, w, yerr=id_stds, label="ID", color="#4c72b0", capsize=4)
    ax.bar(x + w / 2, ood_means, w, yerr=ood_stds, label="OOD", color="#c44e52", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Return")
    ax.set_title("Exp-2: Mode Robustness (S1) — ID vs OOD")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── Mode Responsibilities Heatmap ────────────────────────────────────────────


def plot_responsibilities_heatmap(
    results: List[Dict[str, Any]],
    output_path: Path,
    title: str = "Mode Responsibilities over Bridge Steps",
) -> None:
    """Heatmap of w_c(k) across bridge steps for the first seed."""
    if not results:
        return
    meta = results[0].get("result", {}).get("mode_responsibilities", [])
    if not meta:
        return
    resp = np.array(meta)  # shape (K, num_modes)
    if resp.ndim != 2:
        return

    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(resp.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xlabel("Bridge step k")
    ax.set_ylabel("Mode c")
    ax.set_yticks(range(resp.shape[1]))
    ax.set_yticklabels([f"c={i}" for i in range(resp.shape[1])])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="w_c")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── Fidelity Usage Histogram ─────────────────────────────────────────────────


def plot_fidelity_histogram(
    results: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Stacked bar: how many steps spent at each fidelity level."""
    if not results:
        return
    meta = results[0].get("result", {}).get("fidelity_history", [])
    if not meta:
        # Try from bridge_history
        bh = results[0].get("result", {}).get("bridge_history", [])
        if bh:
            meta = [step.get("fidelity_level", 0) for step in bh if isinstance(step, dict)]
    if not meta:
        return

    levels = sorted(set(meta))
    counts = [meta.count(l) for l in levels]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors = ["#4c72b0", "#55a868", "#c44e52"]
    ax.bar(
        [f"Level {l}" for l in levels],
        counts,
        color=colors[: len(levels)],
    )
    ax.set_ylabel("Number of steps")
    ax.set_title("Fidelity Level Usage (S3)")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Plot co-design experiment results")
    parser.add_argument("--results-root", type=str, default="results", help="Root results directory")
    parser.add_argument("--output-dir", type=str, default="figures/co_design", help="Output directory for figures")
    args = parser.parse_args()

    root = Path(args.results_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Exp-1: S3 Multi-Fidelity ──
    s3_methods = {}
    for name, subdir in [
        ("Single-Fidelity", "phase_c/single_fidelity"),
        ("MF-Geometric", "phase_c/multi_fidelity_geometric"),
        ("MF-Linear", "phase_c/multi_fidelity_linear"),
    ]:
        res = load_results(root / subdir)
        if res:
            s3_methods[name] = res
    if s3_methods:
        plot_return_vs_wallclock(s3_methods, out / "exp1_s3_efficiency.pdf")

    # ── Exp-2: S1 Mode Robustness ──
    id_methods, ood_methods = {}, {}
    for name, id_dir, ood_dir in [
        ("No-Mode", "phase_b/nomode_id", "phase_b/nomode_ood"),
        ("S1 (4 modes)", "phase_b/s1_id", "phase_b/s1_ood"),
    ]:
        id_res = load_results(root / id_dir)
        ood_res = load_results(root / ood_dir)
        if id_res:
            id_methods[name] = id_res
        if ood_res:
            ood_methods[name] = ood_res
    if id_methods or ood_methods:
        plot_id_vs_ood(id_methods, ood_methods, out / "exp2_s1_robustness.pdf")

    # ── Mode Responsibilities ──
    s1_id_res = load_results(root / "phase_b/s1_id")
    if s1_id_res:
        plot_responsibilities_heatmap(
            s1_id_res, out / "mode_responsibilities_id.pdf",
            title="Mode Responsibilities (ID, crawling_ground)",
        )
    s1_ood_res = load_results(root / "phase_b/s1_ood")
    if s1_ood_res:
        plot_responsibilities_heatmap(
            s1_ood_res, out / "mode_responsibilities_ood.pdf",
            title="Mode Responsibilities (OOD, crawling_desert)",
        )

    # ── Fidelity Histogram ──
    mf_res = load_results(root / "phase_c/multi_fidelity_geometric")
    if mf_res:
        plot_fidelity_histogram(mf_res, out / "fidelity_usage.pdf")

    # ── Full comparison (Phase D + E) ──
    full_methods = {}
    for name, subdir in [
        ("MRMFMBD (ID)", "phase_d/full_id"),
        ("MRMFMBD (OOD)", "phase_d/full_ood"),
        ("CMA-ES (ID)", "phase_e/cmaes_id"),
        ("CMA-ES (OOD)", "phase_e/cmaes_ood"),
    ]:
        res = load_results(root / subdir)
        if res:
            full_methods[name] = res
    if full_methods:
        plot_return_vs_wallclock(full_methods, out / "full_comparison.pdf")

    print(f"Done. Figures saved to {out}/")


if __name__ == "__main__":
    main()
