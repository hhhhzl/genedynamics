"""Paper-style appendix figures and a narrow preview from saved PegInsert runs.

Use --appendix and/or --main-preview with --data-root for the frozen 240-run
paper dataset. Canonical mode rejects the historical composite protocol.
The old guided-only draft is available explicitly through --legacy. No solver
is rerun and no executed trajectory is represented as a candidate sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SUITES = (("id_wide", "Nominal"), ("ood_pose", "Pose Mismatch"),
          ("ood_sensing", "Sensing Mismatch"))
METHODS = (("baseline/standalone_rl", "Standalone RL"),
           ("ablation/no_rl_prior", "MGA w/o RL prior"),
           ("main/mga", "MGA"))
BLUE = "#2275a5"
TEAL = "#168f92"
RED = "#d95653"
GRAY = "#9b9fa4"
PAPER_METHODS = (
    ("main/mga", "MGA"), ("ablation/no_rl_prior", "w/o RL prior"),
    ("ablation/no_retraction", "w/o LRC"), ("baseline/issa", "ISSA"),
    ("baseline/atacom", "ATACOM"), ("baseline/mppi", "MPPI"),
    ("baseline/dial", "DIAL"), ("baseline/pegasusflow", "PegasusFlow"),
)
PAIR = ("main/mga", "ablation/no_rl_prior", "ablation/no_retraction")
PREVIEW_SUITE = "ood_pose"
EXPECTED_PROTOCOL = "mga_peg_insert_core_only"


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


def save(fig, output, name, *, fixed_canvas=False):
    options = {} if fixed_canvas else {"bbox_inches": "tight", "pad_inches": .06}
    fig.savefig(output / f"{name}.pdf", dpi=400, **options)
    fig.savefig(output / f"{name}.png", dpi=300, **options)
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


def paper_palette():
    source = ROOT / "latex/latex_mga/tex/paper_colors.tex"
    values = dict(re.findall(r"\\definecolor\{([^}]+)\}\{HTML\}\{([0-9A-Fa-f]{6})\}",
                             source.read_text()))
    palette = {key: "#" + value for key, value in values.items()}
    palette["MPPI"] = "#5C8F2E"  # Same companion green as Surface/H1.
    palette["ISSA"] = "#3A4655"  # Same slate as the existing Surface panel.
    return palette


def paper_style():
    style()
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans", "font.size": 7,
        "axes.labelsize": 7, "axes.titlesize": 8,
        "xtick.labelsize": 6, "ytick.labelsize": 6,
        "text.color": "#3A4655", "axes.labelcolor": "#3A4655",
        "axes.edgecolor": "#52627A", "xtick.major.size": 2,
        "ytick.major.size": 2, "axes.linewidth": .5,
    })


def relpath(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def validate_frozen_result(result, path, suite, seed):
    """Reject mixed historical records without imposing a proposal-mode label."""
    config = result["config_snapshot"]
    metadata = config["metadata"]
    contract = metadata.get("safety_metric_contract", {})
    if (int(result["seed"]) != seed or result["suite"] != suite
            or metadata.get("protocol") != EXPECTED_PROTOCOL
            or metadata.get("protocol_status") != "frozen_core_only"
            or metadata.get("expected_formal_task_runs") != 240
            or config.get("n_steps") != 64 or config.get("device") != "cpu"
            or config["method_params"].get("Nsample") != 64
            or contract.get("primary") != "safe_insertion_success"
            or contract.get("execution_window_steps") != 64
            or contract.get("definition") != "insertion_success_and_no_violation_anywhere_in_64_step_execution"
            or config["method_params"].get("learned_reliability", False)):
        raise ValueError(f"Not a matching frozen 240-run CPU record: {path}")


def result_records(data_root):
    """Read exactly eight methods, three suites, and seeds 0--9 from one root."""
    records = {}
    for suite, _ in SUITES:
        for method, _ in PAPER_METHODS:
            records[method, suite] = []
            for seed in range(10):
                path = data_root / method / f"level_{suite}/seed_{seed}/results.json"
                result = json.loads(path.read_text())
                validate_frozen_result(result, path, suite, seed)
                records[method, suite].append((result, path))
    return records


def outcome(result):
    m = result["metrics"]["peg_insert_metrics"]
    return (0 if m["safe_insertion_success"] > .5 else
            1 if m["insertion_success"] > .5 else 2)


def mean_sd(values):
    return {"n": len(values), "mean": float(np.mean(values)) if values else None,
            "sd_population": float(np.std(values)) if values else None}


def summarize_records(records, data_root):
    stats = {}
    for (method, suite), entries in records.items():
        metrics = [r["metrics"]["peg_insert_metrics"] for r, _ in entries]
        successes = [m["completion_time"] for m in metrics if m["insertion_success"] > .5]
        # The metric extractor saves post-step states, but historical
        # first_event_time uses index*dt. Record both conventions explicitly.
        physical = [m["completion_time"] + r["config_snapshot"]["env_params"].get("dt", .02)
                    for (r, _), m in zip(entries, metrics) if m["insertion_success"] > .5]
        stats.setdefault(suite, {})[method] = {
            "n": 10, "seeds": list(range(10)),
            "outcome_by_seed": [outcome(r) for r, _ in entries],
            "outcome_counts": {label: sum(outcome(r) == i for r, _ in entries)
                               for i, label in enumerate(("safe_completion", "unsafe_completion", "incomplete"))},
            "raw_success": mean_sd([m["insertion_success"] for m in metrics]),
            "safe_success": mean_sd([m["safe_insertion_success"] for m in metrics]),
            "rho_cvar95": mean_sd([m["rho_cvar95"] for m in metrics]),
            "peak_lateral_force_N": mean_sd([m["peak_lateral_force"] for m in metrics]),
            "peak_axial_force_N": mean_sd([m["peak_axial_force"] for m in metrics]),
            "force_torque_violation_rate": mean_sd([m["force_torque_violation_rate"] for m in metrics]),
            "jam_rate": mean_sd([m["jam_rate"] for m in metrics]),
            "successful_completion_time_saved_index_seconds": mean_sd(successes),
            "successful_completion_time_physical_seconds": mean_sd(physical),
            "historical_completion_time_timeout_inclusive_seconds": mean_sd([m["completion_time"] for m in metrics]),
            "recorded_protocols": sorted({r["config_snapshot"]["metadata"]["protocol"] for r, _ in entries}),
            "recorded_prior_modes": sorted({r["config_snapshot"]["method_params"].get("prior_mode", "not_applicable")
                                             for r, _ in entries}),
            "result_paths": [relpath(path) for _, path in entries],
            "result_sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for _, path in entries],
        }
    return {
        "source_root": relpath(data_root),
        "protocol": EXPECTED_PROTOCOL, "run_count": sum(len(v) for v in records.values()),
        "seeds": list(range(10)), "statistics": stats,
        "outcome_figure_methods": [method for method, _ in PAPER_METHODS],
        "definitions": {
            "outcome": "Safe completion = saved full-record SSR; unsafe completion = raw completion with a wrench-limit violation or jam anywhere in the record; incomplete = no raw completion. Categories partition all ten trials.",
            "strict_safe_success": "Insertion success with neither force_torque_violation nor jammed in any of the 64 execution samples. Prefix-safe completion is a separate legacy diagnostic and is not used for outcomes.",
            "rho": "Maximum of true lateral force/20 N, axial force/30 N, bending torque/1.5 N m, and torsional torque/1 N m.",
            "time_axis": "Physical post-step sample time: (sample index + 1)*dt. Initial state is t=0; all 64 execution samples are retained through 1.28 s.",
            "completion_time": "Successful-trial means exclude failures. Historical saved completion_time uses index*dt and substitutes 1.28 s for failure; physical completion time adds one control period to a successful saved index time.",
            "sd": "Population standard deviation across trials, matching canonical aggregate convention.",
        },
        "limitations": [
            "All figures use one complete frozen CPU root; no historical runs or nominal substitutions are pooled.",
            "Learned reliability is disabled in this protocol.",
            "No per-candidate node or rollout geometry is logged. Decision scalars are not candidate trajectories.",
            "No controller experiment was run to generate these figures.",
        ],
    }


def bordered_legend(fig, handles, *, y, ncol, fontsize=5.7):
    legend = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, y),
                        ncol=ncol, frameon=True, fontsize=fontsize, handlelength=2,
                        columnspacing=1.1, borderpad=.55, fancybox=False)
    legend.get_frame().set_edgecolor("#9A9A9A")
    legend.get_frame().set_linewidth(.5)
    legend.get_frame().set_alpha(1)


def paint_socket(image):
    """Match the Figure 2 socket: the same light teal, with the original shading."""
    array = np.asarray(image)
    scale = 255.0 if array.dtype != np.uint8 and np.max(array) <= 1 else 1.0
    rgb = np.clip(array[..., :3].astype(np.float32) * scale, 0, 255)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    socket = (blue > 150) & ((blue - red) > 85) & (green < 125) & (red < 90)
    if not np.any(socket):
        return array
    teal = np.array([176, 214, 210], np.float32)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    painted = np.clip(teal * (luminance / float(luminance[socket].mean()))[..., None], 0, 255)
    rgb[socket] = painted[socket]
    if array.shape[-1] == 4:
        painted_image = array.copy()
        painted_image[..., :3] = rgb.astype(array.dtype) if array.dtype == np.uint8 else rgb / scale
        return painted_image
    if array.dtype == np.uint8:
        return rgb.astype(np.uint8)
    return rgb / scale


def outcome_figure(records, output, colors):
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.15), sharey=True)
    chart_methods = PAPER_METHODS
    fills = [colors["MGATeal"], colors["MGAOrange"], "#D5DAE0"]
    for ax, (suite, title) in zip(axes, SUITES):
        for row, (method, _) in enumerate(chart_methods):
            bottom = 0
            for category, fill in enumerate(fills):
                count = sum(outcome(r) == category for r, _ in records[method, suite])
                if count:
                    ax.bar(row, count, bottom=bottom, width=.76, color=fill,
                            edgecolor="white", linewidth=.4)
                    ax.text(row, bottom + count / 2, str(count), ha="center", va="center",
                            fontsize=6.2, color="white" if category < 2 else "#3A4655")
                bottom += count
        ax.set(ylim=(0, 10.6), yticks=[0, 5, 10])
        ax.set_xticks(range(len(chart_methods)),
                      ["MGA", "w/o prior", "w/o LRC", "ISSA", "ATACOM", "MPPI", "DIAL", "PF"],
                      rotation=55, ha="right")
        ax.set_title(title, fontweight="bold", pad=6)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=".91", lw=.5)
    axes[0].set_ylabel("Trials (n = 10)")
    bordered_legend(fig, [Patch(facecolor=c, label=l) for c, l in zip(fills,
                    ["Safe completion", "Unsafe completion", "Incomplete"])], y=.008, ncol=3)
    fig.subplots_adjust(left=.06, right=.99, top=.85, bottom=.36, wspace=.13)
    save(fig, output, "peg_outcome_decomposition", fixed_canvas=True)


def validated_signals(run):
    result, trajectory, _, _ = run
    signals = trajectory["task_signals"]
    n = len(trajectory["actions"])
    dt = float(signals["dt"])
    if n != 64 or len(trajectory["states"]) != 65 or not np.isclose(dt, .02):
        raise ValueError("Canonical figures require all 64 execution samples at 0.02 s")
    for key in ("rho", "insertion_depth", "success", "strict_safe_success"):
        if len(signals[key]) != n or not np.isfinite(signals[key]).all():
            raise ValueError(f"Invalid execution signal {key}")
    rho = np.maximum.reduce([np.asarray(signals[k]) / d for k, d in
                             (("lateral_force", 20), ("axial_force", 30),
                              ("bending_torque", 1.5), ("torsional_torque", 1))])
    if not np.allclose(rho, signals["rho"], atol=1e-6):
        raise ValueError("Saved rho differs from true wrench utilization")
    metrics = result["metrics"]["peg_insert_metrics"]
    full_safe = (np.any(np.asarray(signals["success"]) > .5)
                 and not np.any(np.asarray(signals["force_torque_violation"]) > .5)
                 and not np.any(np.asarray(signals["jammed"]) > .5))
    if full_safe != bool(metrics["safe_insertion_success"] > .5):
        raise ValueError("Saved SSR differs from the full 64-step safety contract")
    if not np.isclose(np.mean(np.sort(rho)[-4:]), metrics["rho_cvar95"], atol=1e-6):
        raise ValueError("Saved tail metric differs from the highest four utilization samples")
    hit = np.flatnonzero(np.asarray(signals["success"]) > .5)
    if hit.size and not np.isclose(hit[0] * dt, result["metrics"]["peg_insert_metrics"]["completion_time"]):
        raise ValueError("Completion metric index convention changed")
    return signals, (np.arange(n) + 1) * dt, (int(hit[0]) if hit.size else None)


def paired_trace_figure(groups, output, colors, seed):
    fig, axes = plt.subplots(2, 3, figsize=(7, 3.25), sharex=True, sharey="row")
    max_rho = max(max(run[1]["task_signals"]["rho"])
                  for runs in groups.values() for run in runs)
    method_styles = [("main/mga", colors["MGATeal"], "-"),
                     ("ablation/no_rl_prior", colors["MGAIndigo"], "--"),
                     ("ablation/no_retraction", colors["MGAOrange"], ":")]
    for col, (suite, title) in enumerate(SUITES):
        for method, color, ls in method_styles:
            for run in groups[method, suite]:
                signals, time, hit = validated_signals(run)
                emphasized = int(run[0]["seed"]) == seed
                for row, key, scale in ((0, "insertion_depth", 1000), (1, "rho", 1)):
                    vals = np.asarray(signals[key]) * scale
                    axes[row, col].plot(time, vals, color=color, ls=ls,
                                       lw=1.25 if emphasized else .55,
                                       alpha=1 if emphasized else .24,
                                       zorder=3 if emphasized else 2)
                    if row == 0:
                        marker = "o" if outcome(run[0]) == 0 else "x" if hit is not None else "|"
                        idx = hit if hit is not None else len(vals) - 1
                        axes[row, col].plot(time[idx], vals[idx], marker=marker, color=color,
                                           ms=3, markerfacecolor="white", markeredgewidth=.7,
                                           alpha=.95 if emphasized else .5, zorder=5)
        axes[0, col].axhline(32, color="#9A9A9A", lw=.7, ls=":")
        axes[1, col].axhline(1, color=colors["MGAOrangeInk"], lw=.8, ls=":")
        axes[0, col].set_title(title, fontweight="bold")
        for ax in axes[:, col]:
            ax.grid(axis="y", color=".93", lw=.5)
            ax.set(xlim=(0, 1.28), xticks=[0, .4, .8, 1.2])
        axes[1, col].set_xlabel("Physical time (s)")
    axes[0, 0].set(ylabel="Insertion depth (mm)", ylim=(-1, 42), yticks=[0, 16, 32, 40])
    axes[1, 0].set(ylabel=r"Wrench utilization $\rho$", ylim=(0, max(1.3, 1.03 * max_rho)), yticks=[0, .5, 1])
    handles = [Line2D([], [], color=c, ls=ls, lw=1.5, label=l)
               for (_, c, ls), l in zip(method_styles, ["MGA", "w/o RL prior", "w/o LRC"])]
    handles += [Line2D([], [], marker=m, color="#52627A", ls="none", ms=3,
                      markerfacecolor="white", label=l) for m, l in
                [("o", "Safe completion"), ("x", "Unsafe completion"), ("|", "Incomplete endpoint")]]
    bordered_legend(fig, handles, y=.055, ncol=3, fontsize=6)
    fig.subplots_adjust(left=.09, right=.985, top=.92, bottom=.28, wspace=.15, hspace=.22)
    save(fig, output, "peg_paired_depth_rho", fixed_canvas=True)


def decision_rows(run):
    infos = run[1]["infos"]
    if any("additive_prior_selected" not in info for info in infos):
        raise ValueError("Preview requires the recorded prior-selection diagnostic")
    return {
        "additive_prior_selected": [int(info["additive_prior_selected"] > .5) for info in infos],
        "refined_not_revalidated_safe": [int(info["refined_revalidated_safe"] <= .5) for info in infos],
        "emergency_selected": [int(info["emergency_selected"] > .5) for info in infos],
        "selected_revalidated_safe": [int(info["selected_revalidated_safe"] > .5) for info in infos],
    }


def validated_scene_metadata(mga, scene_dir, seed, suite):
    scene_metadata = json.loads((scene_dir / "metadata.json").read_text())[f"peg_{suite}"]
    source = Path(scene_metadata["trajectory"])
    source = source if source.is_absolute() else ROOT / source
    if (scene_metadata["seed"] != seed or source.resolve() != mga[3].resolve()
            or scene_metadata["trajectory_sha256"] != hashlib.sha256(mga[3].read_bytes()).hexdigest()):
        raise ValueError("Native scene metadata does not match the illustrated trajectory")
    return scene_metadata


def native_event_images(mga, scene_dir, seed, suite=PREVIEW_SUITE):
    """Verify actual image-state provenance instead of inferring it from names."""
    signals, time, hit = validated_signals(mga)
    contact = np.flatnonzero(np.asarray(signals["in_contact"]) > .5)
    if not contact.size:
        raise ValueError("The illustrated execution has no recorded contact")
    frame_info = [("initial", "Initial", 0),
                  ("first_contact", "Contact", int(contact[0]) + 1),
                  ("first_completion", "Complete" if hit is not None else "Final state",
                   hit + 1 if hit is not None else len(mga[1]["states"]) - 1)]
    scene_metadata = validated_scene_metadata(mga, scene_dir, seed, suite)
    rendered_events = {frame["image"]: frame
                       for frame in scene_metadata["event_closeups"]["frames"]}
    images = []
    for suffix, label, state_index in frame_info:
        path = scene_dir / f"peg_{suite}_closeup_{suffix}.png"
        if not path.exists():
            raise FileNotFoundError(f"Native Peg preview closeup required: {path}")
        actual = rendered_events[path.name]
        if (actual["state_index"] != state_index or actual["image"] != path.name
                or actual["image_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()):
            raise ValueError(f"Native frame provenance mismatch: {path}")
        images.append((path, label, state_index))
    return images


def rollout_gallery(groups, output, scene_dir, seed):
    """Compose the three native recorded rollouts when all suites are supplied."""
    scene_metadata = json.loads((scene_dir / "metadata.json").read_text())
    if not all(f"peg_{suite}" in scene_metadata for suite, _ in SUITES):
        return
    fig, axes = plt.subplots(3, 5, figsize=(12, 7))
    fig.subplots_adjust(left=.015, right=.995, top=.88, bottom=.055,
                        hspace=.34, wspace=.018)
    fig.suptitle("Peg insertion | MGA", x=.015, y=.975, ha="left",
                 fontsize=16, fontweight="bold")
    frames = []
    for row, ((suite, _), title) in enumerate(zip(SUITES, ["ID", "Pose-OOD", "Sensing-OOD"])):
        mga = next(run for run in groups["main/mga", suite] if int(run[0]["seed"]) == seed)
        metadata = validated_scene_metadata(mga, scene_dir, seed, suite)
        recorded = {frame["image"]: frame for frame in metadata["closeup"]["frames"]}
        dt = float(mga[1]["task_signals"]["dt"])
        for column, state_index in enumerate([0, 16, 32, 48, 64]):
            path = scene_dir / f"peg_{suite}_closeup_frame_{column}.png"
            actual = recorded[path.name]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if (actual["state_index"] != state_index or actual["image_sha256"] != digest
                    or not np.isclose(actual["time_seconds"], state_index * dt)):
                raise ValueError(f"Native gallery frame provenance mismatch: {path}")
            ax = axes[row, column]
            ax.imshow(plt.imread(path))
            ax.axis("off")
            ax.text(.5, -.045, f"t = {state_index * dt:.2f} s", transform=ax.transAxes,
                    ha="center", va="top", fontsize=10)
            frames.append({"suite": suite, "path": relpath(path), "state_index": state_index,
                           "physical_time_seconds": state_index * dt, "image_sha256": digest,
                           "trajectory": relpath(mga[3]), "trajectory_sha256": metadata["trajectory_sha256"]})
        axes[row, 0].text(0, 1.05, title, transform=axes[row, 0].transAxes,
                          fontsize=12, fontweight="bold")
    save(fig, output, "peg_rollouts", fixed_canvas=True)
    (output / "peg_rollouts.json").write_text(json.dumps({
        "method": "main/mga", "seed": seed, "canvas_inches": [12, 7],
        "state_indices": [0, 16, 32, 48, 64], "scene_metadata": relpath(scene_dir / "metadata.json"),
        "definition": "Native saved-state replay only; no policy or dynamics rerun. All frames retain their recorded source-state and image hashes.",
        "frames": frames,
    }, indent=2, allow_nan=False) + "\n")


def geometry_contact_figure(groups, output, scene_dir, colors, seed, data_root):
    suite = PREVIEW_SUITE
    mga = next(run for run in groups["main/mga", suite] if int(run[0]["seed"]) == seed)
    images = native_event_images(mga, scene_dir, seed)
    max_error = max(max(run[1]["task_signals"]["lateral_error"])
                    for method in PAIR[:2] for run in groups[method, suite]) * 1000
    xmax = max(3.1, max_error * 1.06)
    fig = plt.figure(figsize=(7, 3))
    fig.text(.023, .955, "(a) Executed Contact", fontsize=8, fontweight="bold")
    for index, ((path, label, state_index), y) in enumerate(zip(images, [.685, .442, .199])):
        ax = fig.add_axes([.023, y, .20, .223])
        ax.imshow(plt.imread(path))
        ax.axis("off")
        ax.text(.5, .015, f"{label} | {state_index * mga[1]['task_signals']['dt']:.2f} s",
                color="white", fontsize=6, ha="center", va="bottom", transform=ax.transAxes,
                bbox=dict(facecolor="#1D2937", edgecolor="none", alpha=.75, pad=1.8))
    details = []
    for column, (method, title, color, ls) in enumerate([
            ("main/mga", "(b) MGA", colors["MGATeal"], "-"),
            ("ablation/no_rl_prior", "(c) w/o RL Prior", colors["MGAIndigo"], "--")]):
        ax = fig.add_axes([.305 + column * .35, .29, .315, .61])
        ax.add_patch(Rectangle((0, 32), 1.2, 10, facecolor=colors["MGATeal"],
                               alpha=.11, edgecolor="none", zorder=0))
        ax.plot([0, 1.2, 1.2], [32, 32, 42], color=colors["MGATeal"], lw=.75, ls=":")
        ax.text(.46, 37.5, "Position-only\ntarget", ha="center", va="top", fontsize=5.6,
                color=colors["MGATeal"])
        violations = 0
        run_details = []
        for run in sorted(groups[method, suite], key=lambda r: int(r[0]["seed"]) == seed):
            signals, time, hit = validated_signals(run)
            x = np.asarray(signals["lateral_error"]) * 1000
            y = np.asarray(signals["insertion_depth"]) * 1000
            exceeded = np.asarray(signals["rho"]) > 1
            emphasized = int(run[0]["seed"]) == seed
            ax.plot(x, y, color=color, ls=ls, lw=1.35 if emphasized else .65,
                    alpha=1 if emphasized else .28, zorder=3 if emphasized else 2)
            if exceeded.any():
                violations += 1
                points = np.column_stack((x, y))
                segments = np.stack((points[:-1], points[1:]), axis=1)
                ax.add_collection(LineCollection(segments[exceeded[1:]],
                                  colors=colors["MGAOrange"], linewidths=1.7, zorder=6))
                ax.scatter(x[exceeded], y[exceeded], marker="x", s=21,
                           color=colors["MGAOrange"], linewidths=.9, zorder=7)
            ax.plot(x[-1], y[-1], "o", color=color, markerfacecolor="white",
                    markeredgewidth=.6, ms=3.2, alpha=.9, zorder=5)
            if emphasized:
                contact = np.flatnonzero(np.asarray(signals["in_contact"]) > .5)
                for k, marker in [(int(contact[0]), "s"), (hit, "D")]:
                    if k is not None:
                        ax.plot(x[k], y[k], marker, ms=4.5, color=color,
                                markerfacecolor="white", markeredgewidth=1, zorder=8)
            run_details.append({"seed": int(run[0]["seed"]), "trajectory": relpath(run[3]),
                                "safe_success": int(outcome(run[0]) == 0),
                                "raw_success": int(outcome(run[0]) != 2),
                                "violation_sample_count": int(exceeded.sum()),
                                "peak_true_rho": float(max(signals["rho"])),
                                "final_lateral_error_mm": float(x[-1]),
                                "final_depth_mm": float(y[-1])})
        ax.set(xlim=(0, xmax), ylim=(-1, 42), xlabel="Lateral error (mm)", yticks=[0, 16, 32, 40])
        if column == 0:
            ax.set_ylabel("Insertion depth (mm)")
        ax.set_title(title, fontsize=8, fontweight="bold", pad=10)
        ax.grid(axis="y", color=".93", lw=.5)
        details.append({"method": method, "runs": run_details,
                        "trials_with_wrench_exceedance": violations})
    handles = [Line2D([], [], color=colors["MGATeal"], lw=1.4, label="MGA"),
               Line2D([], [], color=colors["MGAIndigo"], ls="--", lw=1.2, label="w/o RL prior"),
               Line2D([], [], color=colors["MGAOrange"], marker="x", ms=4, lw=1,
                      label=r"Wrench limit exceeded ($\rho>1$)"),
               Line2D([], [], color="#52627A", marker="o", markerfacecolor="white", ls="none", ms=3, label="Last recorded state"),
               Line2D([], [], color="#52627A", marker="s", markerfacecolor="white", ls="none", ms=3, label="First contact"),
               Line2D([], [], color="#52627A", marker="D", markerfacecolor="white", ls="none", ms=3, label="First completion")]
    bordered_legend(fig, handles, y=.025, ncol=3, fontsize=6)
    save(fig, output, "peg_geometry_contact", fixed_canvas=True)
    metadata = {
        "suite": suite, "methods": list(PAIR[:2]), "seeds": list(range(10)),
        "source_root": relpath(data_root),
        "image_export_dpi": 400,
        "emphasized_seed": seed, "native_snapshot_seed": seed,
        "selection": "Common fixed seed 0; all ten trajectories shown per method; this example is not selected for maximum separation.",
        "frames": [{"path": relpath(path), "state_index": k,
                    "physical_time_seconds": k * mga[1]["task_signals"]["dt"]}
                   for path, _, k in images],
        "records": details,
        "position_only_target": {"minimum_depth_mm": 32, "maximum_lateral_error_mm": 1.2},
        "warning": "Shading shows necessary positional conditions only. Orientation, wrench limits, jam history, and success hold time also govern safe success. Executed paths are not candidate horizons or a safe set.",
        "exceedance_encoding": "Orange crosses mark each recorded state with true rho>1; orange segments terminate at such states. Circle endpoints are last recorded states, not completion markers.",
        "window": "All64 post-step samples through1.28s, including evolving states after first completion; no smoothing.",
        "caption": "Pose-OOD executed geometry and contact load. Native MGA closeups show the initial pose, first contact, and first completion of the fixed common example (seed 0). Right: executed lateral error versus insertion depth for all 10 runs of MGA and its no-prior ablation; the same example is emphasized. Orange crosses and segments mark true wrench-limit exceedances. Squares and diamonds mark first contact and completion on the emphasized paths; hollow circles are last recorded states. The shaded positional target is necessary but not sufficient for safe success. All records, including post-completion motion, are retained.",
    }
    (output / "peg_geometry_contact.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")


def main_preview(groups, records, output, scene_dir, colors, seed, summary, data_root):
    suite = PREVIEW_SUITE
    paired = {method: next(run for run in groups[method, suite] if int(run[0]["seed"]) == seed)
              for method in PAIR}
    for method in ("baseline/dial", "baseline/issa"):
        result, path = next((r, p) for r, p in records[method, suite] if int(r["seed"]) == seed)
        trajectory_path = path.parent / "trajectory/trajectory.json"
        paired[method] = (result, json.loads(trajectory_path.read_text()), path, trajectory_path)
    mga = paired["main/mga"]
    signals, time, hit = validated_signals(mga)
    images = native_event_images(mga, scene_dir, seed)
    # Match the existing 4.2 x 3.25 inch Surface PDF at the manuscript's
    # 0.64/0.35 widths. The remaining 0.01 of the row is a blank gutter.
    peg_width = 4.2 * .35 / .64
    fig = plt.figure(figsize=(peg_width, 3.25))
    fig.text(.02, .975, "(e) Recorded Contact Progression", fontsize=6.8, fontweight="bold")
    for i, (path, label, index) in enumerate(images):
        ax = fig.add_axes([.018 + i * .328, .755, .315, .205])
        ax.imshow(paint_socket(plt.imread(path)))
        ax.set_axis_off()
        ax.text(.5, -.02, f"{label}  {index * signals['dt']:.2f} s",
                transform=ax.transAxes, ha="center", va="top", fontsize=5.7)
    fig.text(.02, .692, "(f) Executed Wrench Response", fontsize=6.8, fontweight="bold")
    ax = fig.add_axes([.16, .435, .81, .24])
    metadata = []
    curve_styles = [("baseline/issa", colors["ISSA"], (0, (2.5, 1)), .70),
                    ("baseline/dial", colors["MGAOrange"], ":", .8),
                    ("ablation/no_rl_prior", colors["MGAIndigo"], "--", .95),
                    ("main/mga", colors["MGATeal"], "-", 1.45)]
    max_rho = 0
    for method, color, ls, width in curve_styles:
        run = paired[method]
        s, t, done = validated_signals(run)
        max_rho = max(max_rho, max(s["rho"]))
        line, = ax.plot(t, s["rho"], color=color, ls=ls, lw=width,
                        alpha=1 if method == "main/mga" else .70,
                        zorder=5 if method == "main/mga" else 3)
        if method == "main/mga":
            line.set_path_effects([path_effects.Stroke(linewidth=2.6, foreground="white"),
                                   path_effects.Normal()])
        if done is not None:
            ax.plot(t[done], s["rho"][done], "o", ms=3.2, color=color,
                    markerfacecolor="white", markeredgewidth=.8, zorder=6)
        rho = np.asarray(s["rho"])
        contact = np.asarray(s["in_contact"]) > .5
        common_window = (t >= .2 - 1e-6) & (t <= 1.28 + 1e-6)
        metadata.append({"method": method, "seed": seed, "result": relpath(run[2]),
                         "trajectory": relpath(run[3]),
                         "completion_physical_seconds": float(t[done]) if done is not None else None,
                         "min_true_rho": float(min(s["rho"])),
                         "peak_true_rho": float(max(s["rho"])),
                         "peak_physical_time_seconds": float(t[int(np.argmax(rho))]),
                         "margin_to_wrench_limit_at_peak": float(1 - max(rho)),
                         "common_window_contact_audit": {
                             "physical_sample_window_seconds": [.2, 1.28],
                             "sample_count": int(common_window.sum()),
                             "in_contact_samples": int((contact & common_window).sum()),
                             "out_of_contact_samples": int((~contact & common_window).sum()),
                             "zero_rho_samples": int(((rho <= 1e-8) & common_window).sum()),
                             "zero_rho_but_in_contact_samples": int(((rho <= 1e-8) & contact & common_window).sum()),
                             "definition": "Counts of saved post-step states in the common inclusive physical-time window, not an integrated duration or a pooled statistic."},
                         "safe_success": int(outcome(run[0]) == 0),
                         "raw_success": int(outcome(run[0]) != 2),
                         "jam_any": int(np.any(np.asarray(s["jammed"]) > .5)),
                         "comparison_role": "matched_pose_ood_execution",
                         "trajectory_sha256": hashlib.sha256(run[3].read_bytes()).hexdigest()})
    ax.axhline(1, color=colors["MGAOrangeInk"], ls=":", lw=.8)
    ax.text(1.25, 1.01, "Limit", ha="right", va="bottom", fontsize=5.7)
    ax.set(xlim=(0, 1.28), ylim=(0, max(1.12, max_rho * 1.08)), xticks=[0, .4, .8, 1.2], yticks=[0, .5, 1], ylabel=r"True $\rho(t)$")
    ax.tick_params(labelsize=5.7, pad=1)
    ax.yaxis.label.set_size(5.7)
    ax.grid(axis="y", color=".93", lw=.4)
    # A small numerical strip exposes the exact raw peak without smoothing,
    # selecting another seed, or suppressing any baseline excursion.
    fig.add_artist(Rectangle((0.015, 0.342), 0.970, 0.072, transform=fig.transFigure,
                             facecolor="white", edgecolor="#9A9A9A", linewidth=0.55, zorder=2))
    fig.text(.04, .378, "Raw peak\n" + r"$\rho$", fontsize=5.7, va="center", zorder=3)
    example_by_method = {entry["method"]: entry for entry in metadata}
    for x, method, label, color in [
            (.30, "main/mga", "MGA", colors["MGATeal"]),
            (.50, "ablation/no_rl_prior", "w/o prior", colors["MGAIndigo"]),
            (.70, "baseline/dial", "DIAL", colors["MGAOrange"]),
            (.88, "baseline/issa", "ISSA", colors["ISSA"])]:
        fig.text(x, .392, label, ha="center", fontsize=5.7, color=color, zorder=3)
        fig.text(x, .358, f"{example_by_method[method]['peak_true_rho']:.3f}",
                 ha="center", fontsize=5.7, color=color, zorder=3,
                 fontweight="bold" if method == "main/mga" else "normal")
    fig.text(.02, .317, "(g) Logged Plan Selection", fontsize=6.8, fontweight="bold")
    decision = fig.add_axes([.285, .172, .685, .122])
    rows = decision_rows(mga)
    keys = ["additive_prior_selected", "refined_not_revalidated_safe", "emergency_selected"]
    fills = [colors["MGAIndigo"], colors["MGAOrange"], colors["MGAAppendixInk"]]
    dt = float(signals["dt"])
    for row, (key, fill) in enumerate(zip(keys, fills)):
        decision.barh(row, 1.28, height=.66, color="#F1F3F5")
        for step, active in enumerate(rows[key]):
            if active:
                decision.broken_barh([(step * dt, dt)], (row - .33, .66), facecolors=fill)
    decision.set(xlim=(0, 1.28), ylim=(2.55, -.55), xticks=[0, .4, .8, 1.2],
                 yticks=[0, 1, 2], yticklabels=["Prior adopted", "Refined unsafe", "Emergency"],
                 xlabel="Physical time (s)")
    decision.tick_params(axis="y", length=0, pad=3, labelsize=5.7)
    decision.tick_params(axis="x", labelsize=5.7, pad=1)
    decision.xaxis.label.set_size(5.7)
    decision.xaxis.labelpad = 1
    for spine in decision.spines.values():
        spine.set_visible(False)
    handles = [Line2D([], [], color=colors["MGATeal"], label="MGA"),
               Line2D([], [], color=colors["MGAIndigo"], ls="--", label="w/o prior"),
               Line2D([], [], color=colors["ISSA"], ls=(0, (2.5, 1)), label="ISSA"),
               Line2D([], [], color=colors["MGAOrange"], ls=":", label="DIAL"),
               Line2D([], [], marker="o", ms=3, markerfacecolor="white", color="#52627A",
                      ls="none", label="Complete")]
    bordered_legend(fig, handles, y=.008, ncol=3, fontsize=5.7)
    # Preserve the declared narrow canvas instead of content-dependent resizing.
    fig.savefig(output / "peg_insert_preview.pdf", dpi=400)
    fig.savefig(output / "peg_insert_preview.png", dpi=300)
    plt.close(fig)
    summary.update({
        "artifact": "main_panel_preview_only_not_inserted_in_manuscript",
        "canvas_inches": [peg_width, 3.25], "manuscript_width_fractions": [.64, .01, .35],
        "image_export_dpi": 400,
        "illustrated_suite": suite,
        "illustrated_seed": seed, "seed_rule": "Common fixed seed 0; not selected to maximize separation or represent a median.",
        "examples": metadata,
        "curve_display": "All unsmoothed true-rho samples, 64 per method through 1.28 s; no clipping. MGA has a white contrast halo and baseline traces are lighter; values and temporal samples are unchanged. The raw-peak strip reports exact maxima of this common fixed example, not aggregate superiority, stability, or speed. All four traces are matched Pose-OOD executions from the same frozen root. The remaining methods appear in the full outcome comparison.",
        "frame_sources": [{"path": relpath(path), "label": label, "state_index": index,
                           "physical_time_seconds": index * dt} for path, label, index in images],
        "scene_metadata": relpath(scene_dir / "metadata.json"),
        "decision_rows": rows,
        "decision_definition": "Each bar occupies the executed control interval [k*dt,(k+1)*dt]. Prior adopted = logged additive_prior_selected: the accepted expert horizon supplies the next receding control; the full horizon is not executed open-loop. Refined unsafe = the final logged refined_revalidated_safe == 0, not a measured physical violation. Emergency = emergency_selected, not necessarily all candidates failing. Learned reliability is disabled. A successful model certificate is not a guarantee of physical safety.",
        "decision_audit": {
            "prior_adoptions": sum(rows["additive_prior_selected"]),
            "refined_check_failures": sum(rows["refined_not_revalidated_safe"]),
            "refined_failure_steps": [i for i, v in enumerate(rows["refined_not_revalidated_safe"]) if v],
            "failure_context": [{"step": i, "physical_interval_seconds": [i * dt, (i + 1) * dt],
                                 "prior_selected": rows["additive_prior_selected"][i],
                                 "incumbent_revalidated_safe": info.get("incumbent_revalidated_safe"),
                                 "emergency_selected": rows["emergency_selected"][i]}
                                for i, info in enumerate(mga[1]["infos"])
                                if rows["refined_not_revalidated_safe"][i]],
            "emergency_steps": [i for i, v in enumerate(rows["emergency_selected"]) if v],
            "emergency_events": [{"step": i, "physical_interval_seconds": [i * dt, (i + 1) * dt],
                                  "emergency_task_override": info.get("emergency_task_override"),
                                  "incumbent_revalidated_safe": info.get("incumbent_revalidated_safe"),
                                  "refined_revalidated_safe": info.get("refined_revalidated_safe")}
                                 for i, info in enumerate(mga[1]["infos"]) if rows["emergency_selected"][i]],
            "selected_model_certificate_passes": sum(rows["selected_revalidated_safe"]),
            "replans": len(rows["selected_revalidated_safe"]),
            "limitation": "A trace-level audit of which branch supplied the next control, not a controlled demonstration of causality, per-candidate geometry, all safety-gate tests, or universal contact stability."},
        "all_pose_source_adoptions": sum(sum(decision_rows(run)["additive_prior_selected"]) for run in groups["main/mga", suite]),
        "all_pose_replans": sum(len(run[1]["infos"]) for run in groups["main/mga", suite]),
        "no_candidate_geometry": "The figure contains executed data and logged decisions, not candidate clouds, projection trajectories, or fabricated jam events.",
    })
    (output / "peg_insert_preview.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


def combined_preview(output, surface_pdf, pdf_python):
    """Vector-only PDF composition at the actual ICLR 5.5 inch paper width.

    PDF QA/composition uses the separately available PyMuPDF interpreter;
    native Surface contents and source bytes are never rewritten.
    """
    code = r'''
import fitz, hashlib, json, pathlib, sys
surface, peg, output = map(pathlib.Path, sys.argv[1:])
before = hashlib.sha256(surface.read_bytes()).hexdigest()
a, b = fitz.open(surface), fitz.open(peg)
assert len(a) == len(b) == 1
width = 5.5 * 72
left, gap, right = .64 * width, .01 * width, .35 * width
height = a[0].rect.height * left / a[0].rect.width
right_height = b[0].rect.height * right / b[0].rect.width
assert abs(height - right_height) < .002, (height, right_height)
doc = fitz.open()
page = doc.new_page(width=width, height=height)
page.show_pdf_page(fitz.Rect(0, 0, left, height), a, 0)
page.show_pdf_page(fitz.Rect(left + gap, 0, width, height), b, 0)
target = output / 'surface_peg_preview.pdf'
doc.save(target)
page.get_pixmap(matrix=fitz.Matrix(4, 4)).save(output / 'surface_peg_preview.png')
assert before == hashlib.sha256(surface.read_bytes()).hexdigest()
metadata = {'scope': 'Reports-only vector composition; existing Surface unmodified; no manuscript insertion',
 'surface_pdf': str(surface), 'surface_sha256_unchanged': before,
 'peg_pdf': str(peg), 'paper_width_inches': 5.5,
 'canvas_inches': [width / 72, height / 72], 'width_fractions': [.64, .01, .35],
 'scaled_panel_heights_points': [height, right_height]}
(output / 'surface_peg_preview.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(json.dumps(metadata, indent=2))
'''
    subprocess.run([pdf_python, "-c", code, str(surface_pdf),
                    str(output / "peg_insert_preview.pdf"), str(output)], check=True)


def refresh_contact_scenes(args):
    """Refresh only the three native replay images in the current appendix PDF.

    Current vector curves, event timestamps, fonts and layout are authoritative;
    this mode never reruns the chart generator or substitutes historical runs.
    """
    import io
    import tempfile
    import pymupdf
    from PIL import Image

    run_dir = args.data_root / "main/mga" / f"level_{PREVIEW_SUITE}" / f"seed_{args.seed}"
    result_path, trajectory_path = run_dir / "results.json", run_dir / "trajectory/trajectory.json"
    run = (json.loads(result_path.read_text()), json.loads(trajectory_path.read_text()),
           result_path, trajectory_path)
    frames = native_event_images(run, args.scene_dir, args.seed)
    pdf_path = args.figure_pdf or ROOT / "latex/latex_mga/figures/exp/appendix/peg_geometry_contact.pdf"
    document = pymupdf.open(pdf_path)
    if len(document) != 1:
        raise ValueError("Expected the existing one-page Peg geometry figure")
    page = document[0]
    text_before = page.get_text()
    selected = sorted([(item, page.get_image_rects(item[0])[0])
                       for item in page.get_images(full=True)
                       if len(page.get_image_rects(item[0])) == 1], key=lambda entry: entry[1].y0)
    if len(selected) != 3 or any(rect.x1 > page.rect.width * .25 for _, rect in selected):
        raise ValueError("Current PDF no longer has exactly three left-hand replay frames")
    before = page.get_pixmap(matrix=pymupdf.Matrix(3, 3), alpha=False)
    dt = float(run[1]["task_signals"]["dt"])
    for (item, rect), (path, label, index) in zip(selected, frames):
        if f"{label} | {index * dt:.2f} s" not in text_before:
            raise ValueError("Saved event time differs from the current caption; refusing to relabel")
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            if not np.isclose(rgb.width / rgb.height, rect.width / rect.height, atol=1e-5):
                raise ValueError("Replay aspect ratio differs; do not distort saved geometry")
            stream = io.BytesIO()
            rgb.save(stream, format="PNG")
        page.replace_image(item[0], stream=stream.getvalue())
    after = page.get_pixmap(matrix=pymupdf.Matrix(3, 3), alpha=False)
    old_pixels = np.frombuffer(before.samples, np.uint8).reshape(before.height, before.width, before.n)
    new_pixels = np.frombuffer(after.samples, np.uint8).reshape(after.height, after.width, after.n)
    outside = np.ones(old_pixels.shape[:2], bool)
    for _, rect in selected:
        outside[max(0, int(rect.y0*3)-2):int(np.ceil(rect.y1*3))+2,
                max(0, int(rect.x0*3)-2):int(np.ceil(rect.x1*3))+2] = False
    if page.get_text() != text_before or not np.array_equal(old_pixels[outside], new_pixels[outside]):
        raise ValueError("Text or pixels outside replay scenes changed")
    with tempfile.TemporaryDirectory(prefix="mga-peg-scene-refresh-", dir="/private/tmp") as scratch:
        generated = Path(scratch) / pdf_path.name
        document.save(generated, garbage=3, deflate=True)
        document.close()
        generated.replace(pdf_path)
    record = json.loads((args.scene_dir / "metadata.json").read_text())[f"peg_{PREVIEW_SUITE}"]
    record.setdefault("published_figures", {})[str(pdf_path)] = {
        "frames": [{"image": path.name, "state_index": index,
                    "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for path, _, index in frames],
        "camera": record["closeup"]["camera"],
        "pdf_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        "outside_scene_pixel_equality_verified_at_216_dpi": True,
        "data_recomputed": False,
    }
    # Merge only this scene entry; other render jobs may be writing concurrently.
    import importlib.util
    spec = importlib.util.spec_from_file_location("mga_scene_refresh", ROOT / "scripts/paper/mga/render_mechanism_scenes.py")
    scenes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scenes)
    scenes._persist_scene_metadata(args.scene_dir / "metadata.json", {f"peg_{PREVIEW_SUITE}": record})
    print(f"Refreshed replay images only; current curves/text preserved: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        help="Required in canonical mode: one complete frozen 240-run result root")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--appendix", action="store_true", help="Outcome and paired depth/rho appendix figures")
    parser.add_argument("--main-preview", action="store_true", help="Narrow main panel preview; requires native scene PNGs")
    parser.add_argument("--combined-preview", action="store_true", help="Compose existing Surface and Peg PDFs at paper width")
    parser.add_argument("--surface-pdf", type=Path, default=ROOT / "latex/latex_mga/figures/exp/surface_mechanism.pdf")
    parser.add_argument("--pdf-python", default="python3", help="Interpreter with PyMuPDF for vector composition")
    parser.add_argument("--legacy", action="store_true", help="Old guided-only proposal-weight draft; cannot consume additive OOD")
    parser.add_argument("--scene-dir", type=Path, default=ROOT / "reports/mga/paper_figures/appendix/scenes")
    parser.add_argument("--seed", type=int, default=0, choices=range(10))
    parser.add_argument("--refresh-contact-scenes", action="store_true",
                        help="Update only native replay images in the current appendix geometry PDF")
    parser.add_argument("--figure-pdf", type=Path,
                        help="Existing one-page geometry PDF for --refresh-contact-scenes")
    args = parser.parse_args()
    if args.refresh_contact_scenes:
        if args.data_root is None or args.legacy or args.appendix or args.main_preview or args.combined_preview:
            parser.error("Scene refresh requires --data-root and must be separate from chart generation")
        args.data_root = args.data_root.resolve()
        return refresh_contact_scenes(args)
    if args.figure_pdf is not None:
        parser.error("--figure-pdf requires --refresh-contact-scenes")
    if args.legacy and (args.appendix or args.main_preview or args.combined_preview):
        parser.error("Legacy and canonical paper modes must be run separately")
    if not (args.legacy or args.appendix or args.main_preview or args.combined_preview):
        parser.error("Select --appendix, --main-preview, --combined-preview, or explicit --legacy")
    if args.data_root is None:
        if not args.legacy:
            parser.error("Canonical mode requires explicit --data-root; historical data is never substituted")
        args.data_root = ROOT / "results/arm/peg_insert"
    args.data_root = args.data_root.resolve()
    if (args.main_preview or args.appendix) and args.seed != 0:
        parser.error("Native scene assets are fixed to seed 0; other seeds require matched scene assets")
    args.output_dir = args.output_dir or (ROOT / "latex/latex_mga/output/pdf" if args.legacy
                                          else ROOT / "reports/mga/paper_figures")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.legacy:
        paper_style()
        colors = paper_palette()
        records = result_records(args.data_root)
        summary = summarize_records(records, args.data_root)
        groups = {(method, suite): read_runs(args.data_root, method, suite)
                  for method in PAIR for suite, _ in SUITES}
        if args.appendix:
            appendix_dir = args.output_dir / "appendix"
            appendix_dir.mkdir(parents=True, exist_ok=True)
            outcome_figure(records, appendix_dir, colors)
            paired_trace_figure(groups, appendix_dir, colors, args.seed)
            geometry_contact_figure(groups, appendix_dir, args.scene_dir, colors, args.seed, args.data_root)
            rollout_gallery(groups, appendix_dir, args.scene_dir, args.seed)
            (appendix_dir / "peg_additional_metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        if args.main_preview:
            main_preview(groups, records, args.output_dir, args.scene_dir, colors, args.seed, summary, args.data_root)
        if args.combined_preview:
            combined_preview(args.output_dir, args.surface_pdf, args.pdf_python)
        print(json.dumps({"output_dir": str(args.output_dir), "appendix": args.appendix,
                          "main_preview": args.main_preview, "seed": args.seed,
                          "protocol": EXPECTED_PROTOCOL, "run_count": summary["run_count"]}, indent=2))
        return
    style()
    groups = {(method, suite): read_runs(args.data_root, method, suite)
              for method, _ in METHODS for suite, _ in SUITES}
    if any(run[0]["config_snapshot"]["method_params"].get("prior_mode", "guided") != "guided"
           for suite, _ in SUITES for run in groups["main/mga", suite]):
        parser.error("--legacy requires a historical guided-only data root; canonical OOD is additive")
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
