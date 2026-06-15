"""
Cost vs diffusion-denoising-step curves for the 2GO paper benchmarks.

Six figures, one per benchmark, each overlaying the 5 algorithms
(MBD, EB-MBD, MDOC, MD-COAS, 2GO):
  S1     -> stepping stones, level 1
  S5     -> stepping stones, level 5
  zone_a/b/c/d -> humanoid corridor 2D

Curve = per-seed mean over modes, then mean +/- 2*SEM over seeds
(same recipe as results/single2d/visualizations/cost_vs_step).

Run:
  python -m scripts.paper.2go.plot_cost_vs_step
  # or
  python scripts/paper/2go/plot_cost_vs_step.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[3]
STEPPING_ROOT = _ROOT / "results" / "quadruped" / "stepping_stones_2d" / "main"
HUMANOID_ROOT = _ROOT / "results" / "humanoid" / "corridor_2d" / "main"
STEPPING_OUT = _ROOT / "results" / "quadruped" / "stepping_stones_2d" / "vis"
HUMANOID_OUT = _ROOT / "results" / "humanoid" / "corridor_2d" / "vis"

ALGOS_ORDER = ["mbd", "ebmbd", "mdoc", "mdcoas", "twogo"]
ALGO_DISPLAY = {
    "mbd": "MBD",
    "ebmbd": "EB-MBD",
    "mdoc": "MDOC",
    "mdcoas": "MD-COAS",
    "twogo": "2GO",
}
ALGO_COLORS = {
    "mbd": "#CC79A7",
    "ebmbd": "#9467bd",
    "mdoc": "#2ca02c",
    "mdcoas": "#1f77b4",
    "twogo": "#d62728",
}

SEEDS = list(range(5))
STD_BAND_ALPHA = 0.12
UNCERTAINTY_MULTIPLIER = 2.0


def _seed_dir(benchmark: str, algo: str, key: str, seed: int) -> Path:
    """Path to a seed dir. key is the stepping level ('level_1') or humanoid zone ('a'..'d')."""
    if benchmark == "stepping":
        return STEPPING_ROOT / algo / key / f"seed_{seed}"
    return HUMANOID_ROOT / f"{algo}_zone_{key}" / "level_1" / f"seed_{seed}"


def _load_seed_means(benchmark: str, algo: str, key: str):
    """Return (n_seeds, H) array of per-seed mean-over-modes cost curves, or None."""
    seed_means = []
    for seed in SEEDS:
        cost_path = _seed_dir(benchmark, algo, key, seed) / "cost" / "cost.json"
        if not cost_path.exists():
            continue
        with open(cost_path) as f:
            data = json.load(f)
        cpm = data.get("cost_per_mode", [])
        if not cpm:
            continue
        arr = np.asarray(cpm, dtype=np.float64)  # (M, H)
        seed_means.append(np.mean(arr, axis=0))   # (H,)
    if not seed_means:
        return None
    H = min(c.shape[0] for c in seed_means)
    return np.array([c[:H] for c in seed_means])


def _set_diffusion_axis(ax, n_steps: int) -> None:
    tick_positions = [0]
    tick_labels = ["100"]
    for k in [80, 60, 40, 20]:
        pos = n_steps - k
        if 0 < pos < n_steps - 1:
            tick_positions.append(pos)
            tick_labels.append(str(k))
    tick_positions.append(n_steps - 1)
    tick_labels.append("1")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Diffusion Denoising Step", fontsize=20)
    ax.tick_params(axis="x", labelsize=20)
    pad = max(3, int(n_steps * 0.03))
    ax.set_xlim(-pad, n_steps - 1 + pad)


def plot_benchmark(benchmark: str, key: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    n_steps = None
    plotted = False
    for algo in ALGOS_ORDER:
        seed_means = _load_seed_means(benchmark, algo, key)
        if seed_means is None:
            print(f"    [skip] {algo}: no cost data")
            continue
        mean_curve = np.mean(seed_means, axis=0)
        n_seeds = seed_means.shape[0]
        sem_curve = np.std(seed_means, axis=0) / np.sqrt(n_seeds) * UNCERTAINTY_MULTIPLIER
        x = np.arange(len(mean_curve))
        color = ALGO_COLORS[algo]
        ax.fill_between(x, mean_curve - sem_curve, mean_curve + sem_curve, alpha=STD_BAND_ALPHA, color=color)
        ax.plot(x, mean_curve, color=color, linewidth=2.5, label=ALGO_DISPLAY[algo])
        n_steps = len(mean_curve)
        plotted = True
    if not plotted:
        print(f"  [WARN] nothing plotted for {title}")
        plt.close(fig)
        return
    _set_diffusion_axis(ax, n_steps or 100)
    ax.set_ylabel("Cost", fontsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.set_title(title, fontsize=21, fontweight="bold")
    ax.legend(loc="best", fontsize=15)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


def main() -> None:
    STEPPING_OUT.mkdir(parents=True, exist_ok=True)
    HUMANOID_OUT.mkdir(parents=True, exist_ok=True)
    print(f"Stepping output: {STEPPING_OUT}")
    print(f"Humanoid output: {HUMANOID_OUT}")
    jobs = [
        ("stepping", "level_1", "Cost vs Diffusion Step (S1: Stepping Stones L1)", STEPPING_OUT / "s1.png"),
        ("stepping", "level_5", "Cost vs Diffusion Step (S5: Stepping Stones L5)", STEPPING_OUT / "s5.png"),
        ("humanoid", "a", "Cost vs Diffusion Step (Corridor Zone A)", HUMANOID_OUT / "zone_a.png"),
        ("humanoid", "b", "Cost vs Diffusion Step (Corridor Zone B)", HUMANOID_OUT / "zone_b.png"),
        ("humanoid", "c", "Cost vs Diffusion Step (Corridor Zone C)", HUMANOID_OUT / "zone_c.png"),
        ("humanoid", "d", "Cost vs Diffusion Step (Corridor Zone D)", HUMANOID_OUT / "zone_d.png"),
    ]
    for benchmark, key, title, out_path in jobs:
        print(f"Plot: {title}")
        plot_benchmark(benchmark, key, title, out_path)
    print("Done.")


if __name__ == "__main__":
    main()
