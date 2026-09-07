"""Paper-style previews from saved PegInsert records; no simulated samples.

The trajectory figure shows executed rollouts, not proposal trajectories.
The source figure uses the saved reverse-step-averaged source weights, with
deterministic anchors excluded. Results are grouped by their recorded config,
not by the current (possibly different) YAML configuration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SUITES = (("id_wide", "Nominal"), ("ood_pose", "Pose mismatch"),
          ("ood_sensing", "Sensing mismatch"))
METHODS = (("baseline/standalone_rl", "Standalone RL"),
           ("ablation/no_rl_prior", "MGA w/o RL prior"),
           ("main/mga", "MGA"))
BLUE = "#2275a5"
TEAL = "#168f92"
RED = "#d95653"
GRAY = "#9b9fa4"


def read_runs(data_root, method, suite):
    runs = []
    for result_path in sorted((data_root / method / f"level_{suite}").glob(
            "seed_*/results.json")):
        trajectory_path = result_path.parent / "trajectory/trajectory.json"
        result = json.loads(result_path.read_text())
        trajectory = json.loads(trajectory_path.read_text())
        runs.append((result, trajectory, result_path, trajectory_path))
    if not runs:
        raise ValueError(f"No recorded runs for {method}/{suite}")
    if sorted(run[0]["seed"] for run in runs) != list(range(10)):
        raise ValueError(f"Expected exactly seeds 0-9 for {method}/{suite}")
    for result, trajectory, _, _ in runs:
        if method == "main/mga" and len(trajectory["infos"]) != len(trajectory["actions"]):
            raise ValueError("Saved action/diagnostic counts differ")
        if len(trajectory["states"]) != len(trajectory["actions"]) + 1:
            raise ValueError("Saved state/action counts differ")
        params = result["config_snapshot"]["env_params"]
        if not (np.isclose(params.get("success_depth", 0.032), 0.032)
                and np.isclose(params.get("success_lateral_tol", 0.0012), 0.0012)):
            raise ValueError("Position-only target must match the recorded tolerances")
    return runs


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif", "font.size": 8.5,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.7, "lines.linewidth": 1.0,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.facecolor": "white",
    })


def save(fig, output, name):
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(output / f"{name}.png", dpi=210,
                bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def trajectory_figure(groups, output):
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 7.5), sharex=True, sharey=True)
    lateral_max = max(float(np.max(run[1]["task_signals"]["lateral_error"]))
                      for runs in groups.values() for run in runs) * 1000
    depth_max = max(float(np.max(run[1]["task_signals"]["insertion_depth"]))
                    for runs in groups.values() for run in runs) * 1000
    xmax = max(2.5, lateral_max * 1.1)
    ymax = max(42.0, depth_max * 1.04)
    stats = {}
    for row, (suite, suite_title) in enumerate(SUITES):
        stats[suite] = {}
        for col, (method, title) in enumerate(METHODS):
            ax = axes[row, col]
            runs = groups[(method, suite)]
            # These are necessary positional success conditions, NOT the
            # complete feasible set (orientation, contact limits, hold time).
            ax.add_patch(Rectangle((0, 32), 1.2, ymax - 32,
                                   facecolor=BLUE, alpha=0.08, edgecolor="none"))
            ax.plot([0, 1.2, 1.2], [32, 32, ymax], color=BLUE, lw=1.1)
            for result, trajectory, _, _ in runs:
                signals = trajectory["task_signals"]
                lateral = np.asarray(signals["lateral_error"], float) * 1000
                depth = np.asarray(signals["insertion_depth"], float) * 1000
                violation = np.asarray(signals["force_torque_violation"], float) > 0
                if not (lateral.shape == depth.shape == violation.shape):
                    raise ValueError("Task signal lengths differ")
                # The saved protocol executes a fixed-length receding rollout.
                # Keep every recorded step, matching the reported safety metrics.
                points = np.column_stack((lateral, depth))
                segments = np.stack((points[:-1], points[1:]), axis=1)
                colors = np.where(violation[1:], RED, GRAY)
                ax.add_collection(LineCollection(segments, colors=colors,
                                                linewidths=0.85, alpha=0.7))
                ax.scatter(lateral[-1], depth[-1], marker="o", s=13,
                           facecolors="white", edgecolors="0.25", linewidths=0.6,
                           zorder=4)
            metrics = [run[0]["metrics"]["peg_insert_metrics"] for run in runs]
            safe = sum(float(m["safe_insertion_success"]) for m in metrics)
            success = sum(float(m["insertion_success"]) for m in metrics)
            ax.text(0.0, 1.015, f"Safe success: {safe:.0f}/{len(runs)}",
                    transform=ax.transAxes, ha="left", va="bottom", fontsize=7.6)
            ax.set(xlim=(-0.05, xmax), ylim=(-1, ymax))
            ax.grid(axis="y", color="0.92", lw=0.6)
            if row == 0:
                ax.set_title(title, pad=19)
            if col == 0:
                ax.set_ylabel(f"{suite_title}\nInsertion depth (mm)")
            if row == 2:
                ax.set_xlabel("Lateral error (mm)")
            stats[suite][method] = {
                "n": len(runs), "insertion_successes": success,
                "safe_insertion_successes": safe,
                "seeds": [r[0]["seed"] for r in runs],
            }
    handles = [Line2D([0], [0], color=GRAY, lw=1.2, label="Executed trajectory"),
               Line2D([0], [0], color=RED, lw=1.3, label="Force/torque violation"),
               Patch(facecolor=BLUE, alpha=0.2, label="Position-only target region"),
               Line2D([0], [0], marker="o", markersize=4, markerfacecolor="white",
                      markeredgecolor="0.25", linestyle="none", label="Final state")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.97),
               ncol=2, frameon=False, fontsize=8.2, columnspacing=1.5)
    fig.suptitle("PegInsert: execution-space trajectories", y=1.005, fontsize=12)
    fig.text(0.5, 0.01,
             "Seeds 0-9; full recorded rollouts. Executed trajectories are not proposal samples.",
             ha="center", fontsize=8, color="0.3")
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.095,
                        top=0.855, wspace=0.12, hspace=0.23)
    save(fig, output, "mga-peg-insert-trajectories")
    return stats


def proposal_figure(groups, output):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharex=True, sharey=True)
    stats = {}
    for ax, (suite, title) in zip(axes, SUITES):
        runs = groups[("main/mga", suite)]
        ratios, prior_fractions, budgets = [], [], []
        for result, trajectory, _, _ in runs:
            params = result["config_snapshot"]["method_params"]
            if params.get("prior_mode", "guided") != "guided":
                raise ValueError("Source-weight figure requires guided-bank records")
            n_rl = int(params["prior_stochastic_samples"])
            n_gauss = int(params["Nsample"]) - n_rl - int(params.get("prior_atacom_samples", 0))
            budgets.append((n_rl, n_gauss))
            prior_fractions.append(n_rl / (n_rl + n_gauss))
            shares = []
            for info in trajectory["infos"]:
                wg = float(info["proposal_gaussian_weight"])
                wr = float(info["proposal_rl_weight"])
                if wg < 0 or wr < 0 or not np.isfinite(wg + wr):
                    raise ValueError("Invalid source weight")
                shares.append(wr / (wr + wg) if wr + wg > 0 else np.nan)
            ratios.append(np.asarray(shares))
            ax.plot(np.arange(len(shares)), shares, color=GRAY, alpha=0.45, lw=0.65)
        if len(set(budgets)) != 1:
            raise ValueError("Different candidate budgets must not be pooled")
        max_steps = max(len(x) for x in ratios)
        stacked = np.full((len(ratios), max_steps), np.nan)
        for i, row in enumerate(ratios):
            stacked[i, :len(row)] = row
        center = np.nanmedian(stacked, axis=0)
        ax.plot(np.arange(max_steps), center, color=TEAL, lw=1.8)
        baseline = prior_fractions[0]
        ax.axhline(baseline, color=BLUE, lw=1, ls="--")
        ax.set_title(title, pad=8)
        ax.set_xlabel("Replanning step")
        ax.set(ylim=(-0.03, 1.03), xlim=(-1, max_steps))
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(axis="y", color="0.92", lw=0.6)
        means = [float(np.nanmean(x)) for x in ratios]
        stats[suite] = {
            "rl_candidates": budgets[0][0], "gaussian_candidates": budgets[0][1],
            "equal_per_candidate_weight_reference": baseline,
            "per_run_mean_rl_weight_share": means,
            "mean_of_run_means": float(np.mean(means)),
            "fraction_of_steps_above_reference_by_run": [
                float(np.mean(x > baseline)) for x in ratios],
            "recorded_run_count_by_replan": np.isfinite(stacked).sum(axis=0).tolist(),
            "definition": "W_RL / (W_RL + W_G); each source weight was averaged over reverse steps",
        }
    axes[0].set_ylabel("RL weight share\n(stochastic proposals only)")
    handles = [Line2D([0], [0], color=GRAY, lw=1, label="Individual run"),
               Line2D([0], [0], color=TEAL, lw=1.8, label="Median across 10 runs"),
               Line2D([0], [0], color=BLUE, lw=1, ls="--",
                      label="Equal weight per candidate (8/64)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.93),
               ncol=3, frameon=False, fontsize=7.5, columnspacing=1.2)
    fig.suptitle("RL / Gaussian proposal contribution", y=1.035, fontsize=12)
    fig.text(0.5, 0.005,
             "8 RL + 56 Gaussian candidates; anchors excluded. Source weights are averaged over reverse steps.",
             ha="center", fontsize=7.5, color="0.3")
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.19,
                        top=0.755, wspace=0.13)
    save(fig, output, "mga-peg-insert-proposal-weights")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "results/arm/peg_insert")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "latex/latex_mga/output/pdf")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()
    groups = {(method, suite): read_runs(args.data_root, method, suite)
              for method, _ in METHODS for suite, _ in SUITES}
    snapshots = [
        {"path": str(result_path.relative_to(ROOT)),
         "trajectory": str(trajectory_path.relative_to(ROOT)),
         "recorded_prior_mode": result["config_snapshot"]["method_params"].get("prior_mode", "guided"),
         "controller": result["config_snapshot"]["method_params"].get("controller_method")}
        for runs in groups.values() for result, _, result_path, trajectory_path in runs
    ]
    summary = {
        "scope": "Draft visualization of saved execution trajectories and source-level sampler diagnostics; no proposal paths were logged.",
        "execution": trajectory_figure(groups, args.output_dir),
        "proposal": proposal_figure(groups, args.output_dir),
        "sources": snapshots,
    }
    target = args.output_dir / "mga-peg-insert-draft-summary.json"
    target.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "sources"}, indent=2))


if __name__ == "__main__":
    main()
