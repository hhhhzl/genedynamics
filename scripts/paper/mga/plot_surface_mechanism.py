"""Render the Surface Scanning paper panel from canonical saved executions.

No controller or simulation is rerun. Native MuJoCo scene assets are generated
separately by render_mechanism_scenes.py; this script draws measured execution
signals, not hypothetical proposal trajectories. Default statistics use the
same seeds 0--9 as the current manuscript, with seed 0 fixed for all examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from genedynamics.evaluation.metrics import trajectory_path_coverage, _polyline_progress


METHODS = (
    ("baseline/issa", "ISSA"),
    ("baseline/atacom", "ATACOM"),
    ("baseline/mppi", "MPPI"),
    ("baseline/dial", "DIAL"),
    ("baseline/pegasusflow", "PegasusFlow"),
    ("main/mga", "MGA"),
    ("ablation/no_rl_prior", "w/o RL prior"),
    ("ablation/no_retraction", "w/o LRC"),
)
RESPONSE_METHODS = (
    ("main/mga", "MGA", "MGATeal", "-"),
    ("baseline/mppi", "MPPI", "goal", "--"),
    ("baseline/dial", "DIAL", "MGAOrange", ":"),
    ("baseline/pegasusflow", "PegasusFlow", "MGAAppendixInk", "-."),
    ("baseline/atacom", "ATACOM", "MGAIndigo", (0, (4, 1.5, 1, 1.5))),
    ("baseline/issa", "ISSA", "slate", (0, (2.5, 1))),
)


def paper_colors():
    source = ROOT / "latex/latex_mga/tex/paper_colors.tex"
    colors = dict(re.findall(
        r"\\definecolor\{([^}]+)\}\{HTML\}\{([0-9A-Fa-f]{6})\}",
        source.read_text(),
    ))
    palette = {name: "#" + colors[name] for name in
               ("MGATeal", "MGAIndigo", "MGAOrange", "MGAOrangeInk", "MGAAppendixInk")}
    # Companion fills and deployment colors supplied with the manuscript's
    # complete stage palette; do not modify the LaTeX palette source here.
    palette.update(prior_fill="#E3DDEF", mga_fill="#D6EDE8",
                   goal_fill="#E4F1D3", goal_light="#EDF3C9", goal="#5C8F2E",
                   slate="#3A4655", divider="#9A9A9A")
    return palette


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans", "font.size": 6.5,
        "axes.labelsize": 6.5, "axes.titlesize": 6.5,
        "xtick.labelsize": 5.7, "ytick.labelsize": 5.7,  # same size as the framed keys under (b) and (c)
        "axes.linewidth": 0.5, "lines.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.size": 2, "ytick.major.size": 2,
        "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "text.color": "#3A4655", "axes.labelcolor": "#3A4655",
        "axes.edgecolor": "#3A4655", "xtick.color": "#3A4655", "ytick.color": "#3A4655",
    })


def record(root, method, suite, seed, *, trajectory=False):
    path = root / method / f"level_{suite}" / f"seed_{seed}" / "results.json"
    result = json.loads(path.read_text())
    if int(result["seed"]) != seed:
        raise ValueError(f"Seed mismatch: {path}")
    metrics = result["metrics"]["arm_surface_scan_metrics"]
    item = {"path": path, "result": result, "metrics": metrics}
    if trajectory:
        trajectory_path = path.parent / "trajectory/trajectory.json"
        saved = json.loads(trajectory_path.read_text())
        signals = saved["task_signals"]
        for name in ("positions", "reference_path", "coverage_positions",
                     "force", "in_contact", "normal_stiffness", "k_surf"):
            if not np.isfinite(np.asarray(signals[name], float)).all():
                raise ValueError(f"Non-finite {name}: {trajectory_path}")
        computed = trajectory_path_coverage(
            signals["coverage_positions"], signals["reference_path"],
            signals["coverage_valid_mask"], signals["path_tolerance"],
        )
        if not np.isclose(computed, metrics["trajectory_path_coverage"]):
            raise ValueError(f"Saved coverage disagrees with canonical metric: {path}")
        item.update(signals=signals, trajectory_path=trajectory_path)
    return item


def source_path(path):
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def summarize(root, method, suite, seeds):
    rows = [record(root, method, suite, seed) for seed in seeds]
    keys = ("trajectory_path_coverage", "realized_path_completion",
            "force_cvar95", "deformation_cvar95")
    values = {key: np.asarray([r["metrics"][key] for r in rows]) for key in keys}
    values["normalized_force_cvar95"] = np.asarray([
        r["metrics"]["force_cvar95"] / r["result"]["config_snapshot"]["env_params"]["f_max"]
        for r in rows
    ])
    return {
        "seeds": seeds,
        "sources": [source_path(r["path"]) for r in rows],
        "metrics": {key: {"mean": float(x.mean()), "std": float(x.std()),
                          "per_seed": x.tolist()} for key, x in values.items()},
    }


def physical_time(run):
    dt = float(run["result"]["config_snapshot"]["env_params"]["dt"])
    # Extracted task signals are post-action, while states include the reset.
    return (np.arange(len(run["signals"]["positions"])) + 1) * dt


# In-figure labels and the keys under the panels share one size. Panel titles stay larger.
ANNOTATION_SIZE = 5.7


def label(fig, x, y, letter, text):
    fig.text(x, y, f"({letter}) {text}", fontweight="bold", va="bottom", fontsize=6.8)


def draw_scene(fig, scene_dir, seed, run):
    """Time-ordered views of the same curved episode, with no image stretching."""
    scene_metadata = json.loads((scene_dir / "metadata.json").read_text())
    for i in range(4):
        key = f"surface_keyframe_{i}"
        path = scene_dir / f"{key}.png"
        metadata = scene_metadata[key]
        if metadata["seed"] != seed:
            raise ValueError("Simulation seed differs; rerender scene assets")
        for name, path_key in (("trajectory_sha256", "trajectory_path"), ("result_sha256", "path")):
            if metadata[name] != hashlib.sha256(run[path_key].read_bytes()).hexdigest():
                raise ValueError("Simulation source changed; rerender scene assets")
        bounds = [0.015 + .247 * i, .764, .232, .178]
        ax = fig.add_axes(bounds)
        pixels = plt.imread(path)
        image_ratio = pixels.shape[1] / pixels.shape[0]
        frame_ratio = bounds[2] * fig.get_figwidth() / (bounds[3] * fig.get_figheight())
        ax.imshow(pixels, extent=(0, image_ratio, 0, 1), aspect="auto")
        # Isotropic contain in physical page coordinates, not aspect='auto'
        # stretching of the image to a panorama-shaped rectangle.
        if frame_ratio >= image_ratio:
            pad = (frame_ratio - image_ratio) / 2
            ax.set(xlim=(-pad, image_ratio + pad), ylim=(0, 1))
        else:
            pad = (image_ratio / frame_ratio - 1) / 2
            ax.set(xlim=(0, image_ratio), ylim=(-pad, 1 + pad))
        ax.set_facecolor(np.mean(pixels[:4, :4, :3], axis=(0, 1)))
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_color("#9A9A9A"); spine.set_linewidth(0.35)
        times = metadata["time_seconds"]
        t = float(times[0] if isinstance(times, list) else times)
        view = ("Overview", "Side", "Top", "Oblique")[i]
        ax.text(0.025, 0.05, f"{view}  {t:.2f} s", transform=ax.transAxes,
                fontsize=ANNOTATION_SIZE, color="white", bbox=dict(facecolor="#233F55", alpha=.8,
                                                       edgecolor="none", pad=1.1))


def surface_projection(runs):
    """Recover the canonical patch and its nominal chart stiffness map.

    Only the display projects EE positions vertically onto the real surface.
    Stored three-dimensional contact-valid coverage is never recomputed in 2-D.
    """
    import jax
    import jax.numpy as jnp
    from genedynamics.core.coverage import surface_geometry as sg
    from genedynamics.core.contact.elastic_foundation import stiffness_field

    run = runs[0]
    cfg = run["result"]["config_snapshot"]["env_params"]
    surf = sg.surface_for_level(cfg["level"], cfg.get("surface_seed", 0))
    xi0 = float(np.asarray(run["signals"]["scan_start"]).reshape(-1)[0])
    xi1 = float(np.asarray(run["signals"]["scan_target"]).reshape(-1)[0])
    offset = .02 + float(cfg.get("standoff", 0.0))
    ref = np.asarray(run["signals"]["reference_path"])
    n0 = np.asarray(sg.normal(surf, xi0, .5)).copy()
    n0 *= np.sign(n0[2] + 1e-9)
    surf = sg.translate(surf, ref[0] - np.asarray(sg.point(surf, xi0, .5)) - offset * n0)
    targets = jax.jit(jax.vmap(lambda x: sg.point(surf, x, .5) + offset * sg.normal(surf, x, .5)))(
        jnp.linspace(xi0, xi1, len(ref)))
    if not np.allclose(np.asarray(targets), ref, atol=2e-6):
        raise ValueError("Reconstructed surface differs from saved reference geometry")
    origin = ref[0]
    xy = [(np.asarray(r["signals"]["positions"]) - origin)[:, :2] * 1000 for r in runs]
    all_xy = np.concatenate(xy + [(ref - origin)[:, :2] * 1000])
    lo, hi = all_xy.min(axis=0) - [4, 3], all_xy.max(axis=0) + [4, 3]
    base, u, v = [np.asarray(p) for p in surf.params[:3]]
    evaluate = jax.jit(jax.vmap(lambda uv: sg.point(surf, uv[0], uv[1])))

    def chart(points):
        points = np.asarray(points)
        world = points / 1000 + origin[:2]
        return np.column_stack(((world[:, 0] - base[0]) / u[0],
                                (world[:, 1] - base[1]) / v[1]))

    def project(points):
        uv = chart(points)
        xyz = (np.asarray(evaluate(jnp.asarray(uv))) - origin) * 1000
        return np.column_stack((xyz[:, 0] + .36 * xyz[:, 1],
                                .20 * xyz[:, 0] + .60 * xyz[:, 1] + .50 * xyz[:, 2]))

    if cfg["medium"] != "hybrid" or surf.kind != "plane":
        raise ValueError("The stiffness-map strip requires the canonical hybrid plane")
    map_keys = ("stiffness_map", "stiffness_map_xi_origin", "stiffness_map_xi_span",
                "stiffness_transition_width", "k_hard", "k_soft")

    def local_stiffness(xi, eta):
        map_xi = jnp.clip((jnp.asarray(xi) - cfg["stiffness_map_xi_origin"])
                         / cfg["stiffness_map_xi_span"], 0., 1.)
        return stiffness_field(cfg["stiffness_map"], map_xi, eta,
                               cfg["k_hard"], cfg["k_soft"],
                               cfg["stiffness_transition_width"])

    for item in runs:
        other = item["result"]["config_snapshot"]["env_params"]
        if any(other[key] != cfg[key] for key in map_keys):
            raise ValueError("Compared runs use different stiffness maps")
        expected = np.asarray(local_stiffness(item["signals"]["scan_xi"], .5))
        if not np.allclose(expected, item["signals"]["k_surf"], atol=.01, rtol=0):
            raise ValueError("Reconstructed stiffness map differs from saved k_surf")

    def material(points):
        uv = chart(points)
        return np.asarray(local_stiffness(uv[:, 0], uv[:, 1])) / 1000

    return xy, lo, hi, project, origin, material


def draw_strip(fig, runs, colors):
    """Eight real hybrid paths over one shared nominal plane stiffness map."""
    # Warm force traces and a light cool material map separate the two physical
    # quantities. Both are tints of the actual manuscript colors, not new hues.
    # Keep one linear force normalization across every method.
    orange = np.asarray(to_rgb(colors["MGAOrange"]))
    cmap = LinearSegmentedColormap.from_list("mga_manuscript_force_warm", [
        (0, .9 * np.ones(3) + .1 * orange),
        (.5, .3 * np.ones(3) + .7 * orange),
        (1, to_rgb(colors["MGAOrangeInk"])),
    ])
    overrange_color = .8 * np.asarray(to_rgb(colors["MGAOrangeInk"]))
    cmap.set_over(overrange_color)
    max_util = max(max(np.asarray(r["signals"]["force"]) / r["signals"]["f_max"])
                   for r in runs)
    norm = Normalize(0, 1.0, clip=False)
    soft_fill = np.asarray(to_rgb(colors["prior_fill"]))
    hard_fill = .85 * np.asarray(to_rgb(colors["mga_fill"])) + .15 * np.asarray(to_rgb(colors["MGATeal"]))
    material_cmap = LinearSegmentedColormap.from_list("mga_surface_stiffness_cool", [soft_fill, hard_fill])
    cfg = runs[0]["result"]["config_snapshot"]["env_params"]
    material_norm = Normalize(cfg["k_soft"] / 1000, cfg["k_hard"] / 1000)
    xy, lo, hi, project, origin, material = surface_projection(runs)
    xlim, ylim = (lo[0], hi[0]), (lo[1], hi[1])
    xedge, yedge = np.linspace(*xlim, 100), np.linspace(*ylim, 8)
    front = project(np.column_stack((xedge, np.full_like(xedge, ylim[0]))))
    right = project(np.column_stack((np.full_like(yedge, xlim[1]), yedge)))
    face = np.vstack((front, right, project(np.column_stack((xedge[::-1], np.full_like(xedge, ylim[1])))),
                      project(np.column_stack((np.full_like(yedge, xlim[0]), yedge[::-1])))))
    lower = face - [0, 2.5]
    projected_paths = [project(p) for p in xy]
    cells_xy = np.asarray([[[left, bottom], [right_x, bottom], [right_x, top], [left, top]]
             for bottom, top in zip(yedge[:-1], yedge[1:])
             for left, right_x in zip(xedge[:-1], xedge[1:])])
    cells = project(cells_xy.reshape(-1, 2)).reshape(-1, 4, 2)
    cell_colors = material_cmap(material_norm(material(cells_xy.mean(axis=1))))
    for i, ((method, title), run, points) in enumerate(zip(METHODS, runs, xy)):
        row, col = divmod(i, 4)
        ax = fig.add_axes([0.016 + .247 * col, .592 if row == 0 else .454,
                           .23, .095])
        ax.add_patch(Polygon(np.vstack([front, front[::-1] - [0, 2.5]]), closed=True,
                             facecolor=colors["mga_fill"], edgecolor=colors["divider"], lw=.35))
        ax.add_patch(Polygon(np.vstack([right, right[::-1] - [0, 2.5]]), closed=True,
                             facecolor=colors["prior_fill"], edgecolor=colors["divider"], lw=.35))
        ax.add_patch(Polygon(face, closed=True, facecolor=colors["mga_fill"],
                             edgecolor=colors["divider"], lw=.45))
        ax.add_collection(PolyCollection(cells, facecolors=cell_colors, edgecolors="face",
                                         linewidth=.05, zorder=1))
        for y in np.linspace(*ylim, 4):
            edge = project(np.column_stack((xedge, np.full_like(xedge, y))))
            ax.plot(*edge.T, color=colors["divider"], alpha=.30, lw=.3, zorder=2)
        for x in (0, (np.asarray(runs[0]["signals"]["reference_path"])[-1, 0]-origin[0])*1000):
            edge = project(np.column_stack((np.full_like(yedge, x), yedge)))
            ax.plot(*edge.T, color=colors["divider"], alpha=.5, lw=.3, zorder=2)
        reference = (np.asarray(run["signals"]["reference_path"]) - origin)[:, :2] * 1000
        ax.plot(*project(reference).T, color=colors["slate"], ls="--", lw=.6, zorder=3)
        displayed = projected_paths[i]
        segments = np.stack([displayed[:-1], displayed[1:]], axis=1)
        contact = np.asarray(run["signals"]["in_contact"]) > 0.5
        util = np.asarray(run["signals"]["force"]) / run["signals"]["f_max"]
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=1.3,
                            capstyle="round", joinstyle="round", zorder=4)
        lc.set_array(util[1:])
        lc.set_path_effects([path_effects.Stroke(linewidth=1.9, foreground="white"),
                             path_effects.Normal()])
        ax.add_collection(lc)
        if (~contact[1:]).any():
            ax.add_collection(LineCollection(segments[~contact[1:]], colors=colors["divider"],
                                             linestyles="dotted", linewidth=1.5, zorder=4))
        ax.scatter(*displayed[-1], s=5, facecolor="white", edgecolor=colors["slate"], lw=.4, zorder=6)
        ax.set(xlim=(face[:, 0].min()-2, face[:, 0].max()+2),
               ylim=(lower[:, 1].min()-1, face[:, 1].max()+5))
        ax.set_axis_off()
        title_color = colors["MGATeal"] if method == "main/mga" else colors["slate"]
        ax.set_title(title, color=title_color, fontsize=ANNOTATION_SIZE,
                     fontweight="bold" if method == "main/mga" else "normal", pad=1)
        coverage = 100 * run["metrics"]["trajectory_path_coverage"]
        ax.text(.97, .01, f"{coverage:.1f}%", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=ANNOTATION_SIZE, color=title_color)
    # One framed key under (b), above the (c) title, same border as the (c) legend.
    # Groups, left to right: nominal stiffness, force scale, reference.
    fig.add_artist(Rectangle((0.012, 0.408), 0.976, 0.040, transform=fig.transFigure,
                             facecolor="white", edgecolor=colors["divider"],
                             linewidth=0.55, zorder=3))
    key_y = 0.428
    fig.text(0.022, key_y, r"Nominal $k$ (kN/m)", fontsize=ANNOTATION_SIZE,
             va="center", ha="left", zorder=5)
    material_ax = fig.add_axes([0.195, 0.421, 0.10, 0.014], zorder=5)
    material_bar = fig.colorbar(plt.cm.ScalarMappable(norm=material_norm, cmap=material_cmap),
                               cax=material_ax, orientation="horizontal")
    material_bar.ax.tick_params(length=0, labelbottom=False)
    material_bar.outline.set_linewidth(.4)
    fig.text(0.305, key_y, "2 (Soft)", fontsize=ANNOTATION_SIZE, va="center", ha="left", zorder=5)
    fig.text(0.375, key_y, "8 (Hard)", fontsize=ANNOTATION_SIZE, va="center", ha="left", zorder=5)
    fig.text(0.475, key_y, r"$F_n/F_{\max}$", fontsize=ANNOTATION_SIZE,
             va="center", ha="left", zorder=5)
    cax = fig.add_axes([0.575, 0.421, 0.10, 0.014], zorder=5)
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation="horizontal", extend="max" if max_util > 1 else "neither")
    bar.ax.tick_params(length=0, labelbottom=False)
    bar.outline.set_linewidth(0.4)
    for x, tick in ((0.685, "0.0"), (0.735, "0.5"), (0.785, "1.0")):
        fig.text(x, key_y, tick, fontsize=ANNOTATION_SIZE, va="center", ha="left", zorder=5)
    fig.add_artist(Line2D([.855, .895], [key_y, key_y], transform=fig.transFigure,
                         color=colors["slate"], ls="--", lw=.65, zorder=5))
    fig.text(.905, key_y, "Ref.", fontsize=ANNOTATION_SIZE, va="center", ha="left", zorder=5)


def response_trend(values):
    """Display-only centered mean; retain both recorded endpoints exactly."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 3 or not np.isfinite(values).all():
        raise ValueError("Response trends require at least three finite samples")
    trend = values.copy()
    trend[1:-1] = (values[:-2] + values[1:-1] + values[2:]) / 3
    return trend


def response_display_report(runs, colors):
    """Save the exact display transform separately from all raw statistics."""
    records = {}
    for (method, title, color_key, ls), run in zip(RESPONSE_METHODS, runs):
        time = physical_time(run)
        force = np.asarray(run["signals"]["force"], float)
        stiffness = np.asarray(run["signals"]["normal_stiffness"], float) / 1000
        peak = int(force.argmax())
        records[method] = {
            "label": title, "color": colors[color_key], "line_style": ls,
            "time_seconds": time.tolist(), "force_raw_N": force.tolist(),
            "force_trend_N": response_trend(force).tolist(),
            "stiffness_raw_kN_per_m": stiffness.tolist(),
            "stiffness_trend_kN_per_m": response_trend(stiffness).tolist(),
            "raw_peak_N": float(force[peak]), "raw_peak_time_seconds": float(time[peak]),
            "raw_force_std_after_0p2s_N": float(force[time > .2].std()),
            "trajectory_path_coverage": run["metrics"]["trajectory_path_coverage"],
        }
    return {"transform": "Identical centered 3-point arithmetic mean for every Fn/Kn trace",
            "endpoints": "Original first/last samples retained; no zero padding or extrapolation",
            "raw_evidence": "Faint unsmoothed traces plus triangles at every raw Fn global maximum",
            "statistics": "Always computed from raw saved records, never from displayed trends",
            "color_and_dash_reference": "plot_humanoid_mechanism.py:FORCE_METHODS",
            "runs": records}


def draw_response(fig, by_method, colors):
    """Same physical clock for MGA and all five plotted external baselines."""
    force_ax = fig.add_axes([.11, .266, .36, .095])
    stiffness_ax = fig.add_axes([.11, .146, .36, .095])
    handles = []
    peak_records = []
    for method, title, color_key, ls in RESPONSE_METHODS:
        color = colors[color_key]
        run = by_method[method]
        time = physical_time(run)
        raw_force = np.asarray(run["signals"]["force"], float)
        raw_stiffness = np.asarray(run["signals"]["normal_stiffness"], float) / 1000
        # The unsmoothed evidence remains visible for every method, especially
        # narrow transients which a moving mean must never silently erase.
        force_ax.plot(time, raw_force, color=color, lw=.38, alpha=.24, zorder=1)
        stiffness_ax.plot(time, raw_stiffness, color=color, lw=.38, alpha=.24, zorder=1)
        line, = force_ax.plot(time, response_trend(raw_force), color=color, ls=ls,
                              lw=1.2 if method == "main/mga" else .8, label=title,
                              zorder=5 if method == "main/mga" else 3)
        handles.append(line)
        stiffness_line, = stiffness_ax.plot(time, response_trend(raw_stiffness),
                          color=color, ls=ls, lw=1.2 if method == "main/mga" else .8,
                          zorder=5 if method == "main/mga" else 3)
        if method == "main/mga":
            for highlighted in (line, stiffness_line):
                highlighted.set_path_effects([
                    path_effects.Stroke(linewidth=1.85, foreground="white", alpha=.85),
                    path_effects.Normal(),
                ])
        peak_index = int(raw_force.argmax())
        force_ax.scatter(time[peak_index], raw_force[peak_index], marker="^", s=7,
                         facecolor=color, edgecolor="white", linewidth=.25, zorder=7)
        peak_records.append((time[peak_index], raw_force[peak_index]))
    mga = by_method["main/mga"]
    time = physical_time(mga)
    force_ax.plot(time, mga["signals"]["force_des"], color=colors["divider"], ls="--", lw=.5)
    force_ax.axhline(mga["signals"]["f_max"], color=colors["slate"], ls=(0, (2, 3)), lw=.5)
    force_ax.set(xlim=(0, time[-1]), ylim=(0, 65), yticks=[0, 30, 60],
                 xticks=[0, 1, 2], ylabel="$F_n$ (N)")
    force_ax.tick_params(labelbottom=False)
    stiffness_ax.set(xlim=(0, time[-1]), ylim=(.35, 1.35), yticks=[.5, 1],
                     xticks=[0, 1, 2], ylabel="$K_n$ (kN/m)", xlabel="Time (s)")
    for ax in (force_ax, stiffness_ax):
        ax.grid(axis="y", color=colors["divider"], alpha=.24, lw=.4)
        ax.yaxis.labelpad = 1.2
        ax.xaxis.labelpad = 1.2
    # Same row-major legend order as H1: MGA / MPPI / DIAL, then PF / ATACOM / ISSA.
    legend_handles = [handles[i] for i in (0, 3, 1, 4, 2, 5)]
    legend_handles.append(Line2D([], [], marker="^", color=colors["slate"], lw=0,
                                 markersize=4.2, label="Raw peak"))
    legend = fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(.29, .071),
               ncol=4, frameon=True, fancybox=False, framealpha=1,
               facecolor="white", edgecolor=colors["divider"],
               fontsize=ANNOTATION_SIZE, handlelength=1.5, labelspacing=.3,
               handletextpad=.35, columnspacing=.7, borderaxespad=0, borderpad=.3)
    legend.get_frame().set_linewidth(.55)
    force_ax.text(.98, .90, "Limit", transform=force_ax.transAxes, ha="right",
                  fontsize=ANNOTATION_SIZE, color=colors["slate"])
    force_ax.annotate("Target", (.98, float(np.asarray(mga["signals"]["force_des"])[-1])),
                      xycoords=force_ax.get_yaxis_transform(), xytext=(0, -2),
                      textcoords="offset points", ha="right", va="top", fontsize=ANNOTATION_SIZE,
                      color=colors["divider"])
    max_time, max_force = max(peak_records, key=lambda pair: pair[1])
    force_ax.annotate("Raw peak", (max_time, max_force), xytext=(5, -1),
                      textcoords="offset points", fontsize=ANNOTATION_SIZE, color=colors["slate"],
                      ha="left" if max_time < time[-1] * .8 else "right", va="center")


def draw_lrc(fig, runs, colors):
    """Restore the executed curved-surface correction evidence, not a proposal sketch."""
    ax = fig.add_axes([.64, .146, .335, .215])
    handles = []
    for run, color, title, ls in zip(runs,
            (colors["MGATeal"], colors["MGAOrange"]), ("MGA", "w/o LRC"), ("-", "--")):
        distance, _ = _polyline_progress(run["signals"]["positions"], run["signals"]["reference_path"])
        line, = ax.plot(physical_time(run), distance * 1000, color=color, ls=ls, lw=1.0, label=title)
        handles.append(line)
    tolerance = float(runs[0]["signals"]["path_tolerance"]) * 1000
    ax.axhspan(0, tolerance, color=colors["goal_fill"], alpha=.6, zorder=0)
    ax.axhline(tolerance, color=colors["goal"], ls=":", lw=.7)
    ax.text(.03, tolerance + .3, "5 mm tolerance", fontsize=ANNOTATION_SIZE, color=colors["goal"])
    ax.set(xlim=(0, physical_time(runs[0])[-1]), ylim=(0, 12.5),
           xticks=[0, 1, 2], yticks=[0, 5, 10], xlabel="Time (s)", ylabel="3-D path error\n(mm)")
    ax.xaxis.labelpad = 1.2
    ax.yaxis.labelpad = 1.2
    ax.grid(axis="y", color=colors["divider"], alpha=.24, lw=.4)
    legend = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.795, .057), ncol=2,
               frameon=True, fancybox=False, framealpha=1,
               facecolor="white", edgecolor=colors["divider"],
               fontsize=ANNOTATION_SIZE, handlelength=1.5, handletextpad=.35,
               columnspacing=.7, borderaxespad=0, borderpad=.35)
    legend.get_frame().set_linewidth(.55)
    coverage = [100 * r["metrics"]["trajectory_path_coverage"] for r in runs]
    ax.text(.04, .94, f"Coverage: {coverage[0]:.1f}% / {coverage[1]:.1f}%",
            transform=ax.transAxes, va="top", fontsize=ANNOTATION_SIZE)


def baseline_selection(root, seeds):
    """Frozen operational SSR ranks current external paper baselines."""
    suites = [p.name.removeprefix("level_") for p in sorted((root / "main/mga").glob("level_*"))
              if (p / "seed_0/results.json").is_file()]
    if len(suites) != 13:
        raise ValueError(f"Expected the frozen 13 surface suites, got {suites}")
    groups = {"policy_based": ["baseline/standalone_rl", "baseline/issa", "baseline/atacom"],
              "external_sampling": ["baseline/mppi", "baseline/dial", "baseline/pegasusflow"]}
    report = {"protocol": "coverage>=.90; settled contact loss<=.10; force violation<=.01",
              "scope": "Current external paper baselines; MBO is an MGA-derived internal variant",
              "seeds": seeds, "suites": suites, "groups": {}}
    for family, methods in groups.items():
        ranking = []
        for method in methods:
            rows = [record(root, method, suite, seed) for suite in suites for seed in seeds]
            ok = [r["metrics"]["trajectory_path_coverage"] >= .90
                  and r["metrics"]["contact_loss_rate_settled"] <= .10
                  and r["metrics"]["force_violation_rate"] <= .01 for r in rows]
            risk = [r["metrics"]["force_cvar95"] / r["result"]["config_snapshot"]["env_params"]["f_max"]
                    for r in rows]
            ranking.append({"method": method, "safe_success": float(np.mean(ok)),
                            "normalized_force_tail": float(np.mean(risk)), "n": len(rows),
                            "sources": [source_path(r["path"]) for r in rows]})
        ranking.sort(key=lambda row: (-row["safe_success"], row["normalized_force_tail"]))
        report["groups"][family] = ranking
    expected = {"policy_based": "baseline/atacom", "external_sampling": "baseline/dial"}
    for group, method in expected.items():
        if report["groups"][group][0]["method"] != method:
            raise ValueError(f"Baseline ranking changed for {group}; review figure selection before exporting")
    return report


def make_figure(runs, scene_run, hybrid_runs, lrc_runs, scene_dir, colors, seed):
    # Left 60% of a landscape 7 x 3.25 in composition. The restored (d)
    # shares the bottom row with (c), rather than adding another tall row.
    fig = plt.figure(figsize=(4.2, 3.25))
    label(fig, .015, .959, "a", "Bumpy-Surface Scanning: Shape and Contact")
    draw_scene(fig, scene_dir, seed, scene_run)
    label(fig, .015, .724, "b", "Hybrid Map and Paths (3-D Contact Coverage)")
    draw_strip(fig, runs, colors)
    label(fig, .015, .377, "c", "Hybrid Response (3-pt Mean)")
    label(fig, .54, .377, "d", "LRC on a Curved Surface")
    draw_response(fig, dict(zip((method for method, *_ in RESPONSE_METHODS), hybrid_runs)), colors)
    draw_lrc(fig, lrc_runs, colors)
    return fig


APPENDIX_SUITES = (
    "rigid_plane", "rigid_cylinder", "rigid_convex", "rigid_bumpy",
    "soft_plane", "soft_cylinder", "soft_convex", "soft_bumpy",
    "hybrid_stripes", "hybrid_center_hard", "hybrid_center_soft",
    "rigid_unseen", "soft_unseen",
)
APPENDIX_METHODS = (
    ("main/mga", "MGA", "MGATeal", "o"),
    ("ablation/no_rl_prior", "w/o RL prior", "MGAIndigo", "^"),
    ("ablation/no_retraction", "w/o LRC", "MGAOrange", "s"),
)


def appendix_stats(values):
    values = np.asarray(values, float)
    if not np.isfinite(values).all():
        raise ValueError("Appendix statistics require finite canonical metrics")
    return {"values": values.tolist(), "mean": float(values.mean()),
            "std": float(values.std(ddof=0)), "std_ddof": 0, "n": len(values)}


def appendix_surface_data(root, seeds):
    """Read only the frozen per-run records; aggregate caches may be stale."""
    rows, sources = [], []
    for method, *_ in APPENDIX_METHODS:
        for suite in APPENDIX_SUITES:
            for seed in seeds:
                run = record(root, method, suite, seed)
                m = run["metrics"]
                env = run["result"]["config_snapshot"]["env_params"]
                row = {"method": method, "suite": suite, "seed": seed,
                       "coverage_percent": 100 * m["trajectory_path_coverage"],
                       "terminal_progress_percent": 100 * m["realized_path_completion"],
                       "normalized_force_cvar95": m["force_cvar95"] / env["f_max"],
                       "deformation_cvar95_mm": 1000 * m["deformation_cvar95"],
                       "settled_contact_loss_rate": m["contact_loss_rate_settled"],
                       "force_violation_rate": m["force_violation_rate"]}
                eligible = (m["trajectory_path_coverage"] >= .9
                            and m["contact_loss_rate_settled"] <= .1)
                row["operational_success"] = eligible and m["force_violation_rate"] <= .01
                row["strict_success"] = eligible and m["force_violation_rate"] == 0
                rows.append(row)
                sources.append({"path": source_path(run["path"]),
                                "sha256": hashlib.sha256(run["path"].read_bytes()).hexdigest()})
    fields = ("coverage_percent", "terminal_progress_percent", "normalized_force_cvar95",
              "deformation_cvar95_mm")
    by_suite, grouped = {}, {}
    for method, *_ in APPENDIX_METHODS:
        by_suite[method] = {}
        for suite in APPENDIX_SUITES:
            selected = [r for r in rows if r["method"] == method and r["suite"] == suite]
            by_suite[method][suite] = {key: appendix_stats([r[key] for r in selected])
                                     for key in fields}
            by_suite[method][suite].update(
                operational_successes=sum(r["operational_success"] for r in selected),
                strict_successes=sum(r["strict_success"] for r in selected))
        grouped[method] = {}
        groups = {"All": APPENDIX_SUITES, "Soft": APPENDIX_SUITES[4:8],
                  "Hybrid": APPENDIX_SUITES[8:11]}
        for group, suites in groups.items():
            selected = [r for r in rows if r["method"] == method and r["suite"] in suites]
            grouped[method][group] = {
                key: appendix_stats([np.mean([r[key] for r in selected if r["seed"] == seed])
                                     for seed in seeds]) for key in fields}
            grouped[method][group].update(
                suites=list(suites), seeds=list(seeds), runs=len(selected),
                operational_successes=sum(r["operational_success"] for r in selected),
                strict_successes=sum(r["strict_success"] for r in selected))
    return {"seeds": seeds, "suite_order": list(APPENDIX_SUITES), "rows": rows,
            "per_suite": by_suite, "grouped": grouped, "sources": sources,
            "aggregation": "Equal suite means within each seed; mean and population SD across seeds",
            "palette_source": "latex/latex_mga/tex/paper_colors.tex",
            "semantics": [
                "Canonical per-run results only; no overall_summary cache and no simulation replay.",
                "Terminal progress projects the final EE point; coverage counts contact-valid reference waypoints visited within 5 mm.",
                "Coverage can exceed terminal progress; these are distinct, non-nested metrics.",
                "Soft excludes soft_unseen; both unseen suites remain visible in the 13-suite plot.",
                "Force and deformation tails are recorded task metrics; hybrid force is analytic Winkler reaction.",
                "no_retraction is the recorded no-LRC ablation, not a physical retract action.",
                "Operational success: coverage >= 90%, settled contact loss <= 10%, force violation <= 1%; strict additionally requires zero force violations.",
            ]}


def save_appendix_surface(fig, output, name, data):
    stem = output / name
    for extension in ("pdf", "png"):
        fig.savefig(stem.with_suffix("." + extension), dpi=300, facecolor="white")
    plt.close(fig)
    stem.with_name(stem.name + "_data").with_suffix(".json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n")
    print(stem.with_suffix(".pdf"))


def plot_surface_appendix(root, output, seeds):
    output.mkdir(parents=True, exist_ok=True)
    data = appendix_surface_data(root, seeds)
    colors = paper_colors()
    style()
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7})
    labels = [s.replace("rigid_", "Hard / ").replace("soft_", "Soft / ")
              .replace("hybrid_", "Hybrid / ").replace("_", " ") for s in APPENDIX_SUITES]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.5), sharey=True)
    fig.subplots_adjust(left=.21, right=.98, top=.88, bottom=.18, wspace=.13)
    offsets = np.linspace(-.17, .17, len(seeds))
    for ax, field, title in zip(axes, ("terminal_progress_percent", "coverage_percent"),
                               ("(a) Terminal Path Progress", "(b) Contact-Valid Coverage")):
        for i, suite in enumerate(APPENDIX_SUITES):
            a = np.asarray(data["per_suite"]["main/mga"][suite][field]["values"])
            b = np.asarray(data["per_suite"]["ablation/no_retraction"][suite][field]["values"])
            if suite == "rigid_unseen":
                ax.axhspan(i - .45, i + .45, color=colors["prior_fill"], zorder=0)
            for x1, x2, dy in zip(a, b, offsets):
                ax.plot([x1, x2], [i + dy, i + dy], color=colors["divider"], alpha=.3, lw=.5)
            for values, key, marker in ((a, "MGATeal", "o"), (b, "MGAOrange", "s")):
                ax.scatter(values, i + offsets, s=9, color=colors[key], alpha=.42, linewidths=0)
                ax.scatter(values.mean(), i, s=31, color=colors[key], marker=marker,
                           edgecolors="white", linewidths=.65, zorder=4)
        for y in (3.5, 7.5, 10.5):
            ax.axhline(y, color=colors["divider"], lw=.45, alpha=.45)
        ax.set(xlim=(-2, 104), xticks=[0, 25, 50, 75, 100], xlabel="Path fraction (%)",
               title=title, ylim=(12.65, -.6))
        ax.set_yticks(range(13)); ax.set_yticklabels(labels)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=5)
        ax.grid(axis="x", color=colors["divider"], alpha=.15, linewidth=.5)
    axes[1].axvline(90, color=colors["slate"], ls=(0, (3, 3)), lw=.6, alpha=.7)
    fig.suptitle("Realized Path Fidelity Across Surface Conditions", x=.21,
                 ha="left", y=.98, fontsize=10, fontweight="bold")
    handles = [Line2D([], [], marker=marker, color=colors[key], lw=0, markersize=5, label=label)
               for _, label, key, marker in (APPENDIX_METHODS[0], APPENDIX_METHODS[2])]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.57, .049),
               ncol=2, frameon=False, fontsize=8)
    fig.text(.21, .025, "Small points: paired seeds. Large markers: suite means. Dashed: 90% coverage.", fontsize=7)
    save_appendix_surface(fig, output, "surface_coverage_progress", data)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.45))
    fig.subplots_adjust(left=.105, right=.97, top=.86, bottom=.17, wspace=.29, hspace=.43)
    fields = ("normalized_force_cvar95", "deformation_cvar95_mm")
    for i, group in enumerate(("Soft", "Hybrid")):
        for j, field in enumerate(fields):
            ax = axes[i, j]
            mga = data["grouped"]["main/mga"][group]
            for method, label_text, key, marker in APPENDIX_METHODS:
                values = data["grouped"][method][group]
                x = np.asarray(values["coverage_percent"]["values"])
                y = np.asarray(values[field]["values"])
                if method != "main/mga":
                    for x0, y0, x1, y1 in zip(mga["coverage_percent"]["values"], mga[field]["values"], x, y):
                        ax.plot([x0, x1], [y0, y1], color=colors["divider"], lw=.55, alpha=.3, zorder=1)
                ax.scatter(x, y, color=colors[key], marker=marker, s=20, alpha=.58,
                           linewidths=.3, edgecolors="white", zorder=3)
                ax.scatter(x.mean(), y.mean(), color=colors[key], marker=marker, s=62,
                           linewidths=.8, edgecolors="white", zorder=4)
            ax.set_title(f"({chr(97 + 2 * i + j)}) {group}: " + ("Contact Load" if j == 0 else "Deformation"), loc="left")
            ax.set(xlabel="Contact-valid coverage (%)", xlim=(48, 102), xticks=[50, 75, 100],
                   ylabel="Force nCVaR95" if j == 0 else "Deformation CVaR95 (mm)")
            ax.grid(alpha=.15, lw=.5)
            ax.margins(y=.18)
    fig.suptitle("Coverage, Contact Load, and Deformation", x=.105, ha="left", y=.98,
                 fontsize=10, fontweight="bold")
    fig.text(.105, .915, f"Suites have equal weight within each seed. Paired comparisons use the same {len(seeds)} seeds.", fontsize=7.3)
    handles = [Line2D([], [], marker=marker, color=colors[key], lw=0, markersize=5,
                      label=label_text) for _, label_text, key, marker in APPENDIX_METHODS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.53, .033),
               ncol=3, frameon=False, fontsize=8)
    fig.text(.105, .023, "Small points show seed averages. Large markers show means. Gray links pair seeds. Better outcomes lie rightward and lower.", fontsize=7)
    save_appendix_surface(fig, output, "surface_contact_tradeoff", data)


def plot_surface_execution_geometry(root, output, scene_dir, seed):
    """Link a native saved-state view to observed EE geometry, not proposals."""
    output.mkdir(parents=True, exist_ok=True)
    suite = "soft_convex"
    runs = [record(root, method, suite, seed, trajectory=True)
            for method in ("main/mga", "ablation/no_retraction")]
    metadata_path = scene_dir / "metadata.json"
    scene_record = json.loads(metadata_path.read_text())["surface_soft_convex"]
    if scene_record["seed"] != seed:
        raise ValueError("Native context must use the same predeclared paired seed")
    for key, path in (("result_sha256", runs[0]["path"]),
                      ("trajectory_sha256", runs[0]["trajectory_path"])):
        if scene_record[key] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("Native surface context is stale relative to the canonical execution")
    frame = scene_record["frames"][3]
    image_path = scene_dir / frame["image"]
    if frame["image_sha256"] != hashlib.sha256(image_path.read_bytes()).hexdigest():
        raise ValueError("Native surface context image hash mismatch")
    ref = np.asarray(runs[0]["signals"]["reference_path"], float)
    if not np.allclose(ref, runs[1]["signals"]["reference_path"]):
        raise ValueError("Paired geometry requires an identical reference path")
    origin = ref[0]
    tolerance = float(runs[0]["signals"]["path_tolerance"])
    colors = paper_colors(); style()
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7})
    fig = plt.figure(figsize=(7.2, 2.85))
    context = fig.add_axes([.012, .23, .285, .62])
    context.imshow(plt.imread(image_path)); context.axis("off")
    fig.text(.014, .93, "(a) Soft-Convex Contact", fontsize=8.2, fontweight="bold")
    path_ax = fig.add_axes([.39, .48, .24, .35])
    coverage_ax = fig.add_axes([.39, .295, .24, .12])
    error_ax = fig.add_axes([.745, .27, .24, .56])
    fig.text(.39, .93, "(b) Executed Path and Visits", fontsize=8.2, fontweight="bold")
    fig.text(.745, .93, "(c) 3-D Path Error", fontsize=8.2, fontweight="bold")
    ref_xy = (ref - origin)[:, :2] * 1000
    reference_arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(ref, axis=0), axis=1))]
    reference_arc = 100 * reference_arc / reference_arc[-1]
    path_ax.plot(ref_xy[:, 0], ref_xy[:, 1], color=colors["slate"], ls=(0, (3, 2)), lw=.8, zorder=2)
    error_ax.axhspan(0, 1000 * tolerance, color=colors["goal_fill"], alpha=.65, zorder=0)
    error_ax.axhline(1000 * tolerance, color=colors["goal"], ls=(0, (3, 3)), lw=.7)
    report = {"suite": suite, "illustrative_seed": seed,
              "selection": "Predeclared common seed, identical to the existing main-figure examples; no per-method selection",
              "context": {"metadata": source_path(metadata_path), "entry": "surface_soft_convex",
                          "image": source_path(image_path), "image_sha256": frame["image_sha256"],
                          "state_index": frame["state_index"], "time_s": frame["time_seconds"],
                          "native_render_provenance": scene_record},
              "origin_world_m": origin.tolist(), "reference_world_m": ref.tolist(),
              "reference_arc_percent": reference_arc.tolist(),
              "path_tolerance_m": tolerance, "methods": {}, "sources": [],
              "palette_source": "latex/latex_mga/tex/paper_colors.tex",
              "semantics": [
                  "Native image is forward-kinematic rendering of one recorded MGA state, not a resimulated episode.",
                  "Panel b is an orthographic world-x/world-y top-view projection of actual saved EE positions, relative to the reference start.",
                  "Faint lines retain every executed EE point; bold segments have contact-valid endpoints within the full 3-D 5 mm reference-polyline tolerance.",
                  "Panel c uses full 3-D closest-polyline distance, not the 2-D projection and not a force/safety violation.",
                  "Canonical coverage remains the contact-valid 3-D reference-waypoint visitation metric; it is not recomputed from the plot projection.",
                  "The two visitation strips use geometric reference arc length, computed from the saved 3-D reference; colored marks are actually visited reference waypoints.",
                  "Time is post-action physical time (i+1)*dt; no scan_xi or command coordinate is used as executed displacement.",
                  "Final circles are the last executed EE samples; no hypothetical correction or candidate trajectory is drawn.",
              ]}
    max_error = 5.0
    for row, (run, key, label_text) in enumerate(zip(runs, ("MGATeal", "MGAOrange"), ("MGA", "w/o LRC"))):
        s = run["signals"]; pos = np.asarray(s["positions"], float)
        distance, _ = _polyline_progress(pos, ref)
        valid = np.asarray(s["coverage_valid_mask"], bool)
        inside = valid & (distance <= tolerance)
        xy = (pos - origin)[:, :2] * 1000
        path_ax.plot(xy[:, 0], xy[:, 1], color=colors[key], alpha=.26, lw=.8)
        segments = np.stack((xy[:-1], xy[1:]), axis=1)
        path_ax.add_collection(LineCollection(segments[inside[:-1] & inside[1:]],
                                             colors=colors[key], linewidths=1.45, zorder=4))
        path_ax.scatter(*xy[-1], color=colors[key], s=23, edgecolors="white", linewidths=.5, zorder=5)
        times = physical_time(run)
        error_ax.plot(times, distance * 1000, color=colors[key], lw=1.25)
        error_ax.scatter(times[-1], distance[-1] * 1000, color=colors[key], s=20,
                         edgecolors="white", linewidths=.5, zorder=5)
        max_error = max(max_error, float(distance.max()) * 1000)
        cover_pos = np.asarray(s["coverage_positions"], float)[valid]
        visited = np.min(np.linalg.norm(ref[:, None, :] - cover_pos[None, :, :], axis=2), axis=1) <= tolerance
        measured = float(np.mean(visited))
        if not np.isclose(measured, run["metrics"]["trajectory_path_coverage"]):
            raise ValueError("Contact-valid waypoint visitation disagrees with canonical coverage")
        coverage_ax.plot([0, 100], [row, row], color=colors["divider"], alpha=.2, lw=3)
        coverage_ax.scatter(reference_arc[visited], np.full(np.sum(visited), row),
                            color=colors[key], marker="s", s=5, linewidths=0, zorder=3)
        report["methods"][label_text] = {
            "coverage_percent": 100 * measured,
            "terminal_progress_percent": 100 * run["metrics"]["realized_path_completion"],
            "positions_world_m": pos.tolist(), "contact_valid": valid.tolist(),
            "within_3d_tolerance_and_contact": inside.tolist(), "reference_visited": visited.tolist(),
            "time_s": times.tolist(), "path_error_3d_mm": (distance * 1000).tolist(),
            "final_error_3d_mm": float(distance[-1] * 1000),
            "max_error_3d_mm": float(distance.max() * 1000)}
        report["sources"].extend({"path": source_path(run[k]), "sha256": hashlib.sha256(run[k].read_bytes()).hexdigest()}
                                 for k in ("path", "trajectory_path"))
    all_xy = np.concatenate([(np.asarray(r["signals"]["positions"]) - origin)[:, :2] * 1000
                             for r in runs] + [ref_xy])
    lo, hi = all_xy.min(axis=0) - [1.5, 1.2], all_xy.max(axis=0) + [1.5, 1.2]
    path_ax.set(xlabel="World x offset (mm)", ylabel="World y offset (mm)", xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]),
                xticks=[0, 10, 20, 30], yticks=[-5, 0])
    path_ax.set_aspect("equal", adjustable="box")
    path_ax.set_anchor("N")
    coverage_ax.set(xlim=(-2, 102), ylim=(1.6, -.6), yticks=[0, 1],
                    yticklabels=["MGA", "w/o LRC"], xticks=[0, 50, 100],
                    xlabel="Reference arc length (%)")
    coverage_ax.tick_params(axis="y", length=0, labelsize=6.6)
    coverage_ax.spines["left"].set_visible(False)
    error_ax.set(xlabel="Time (s)", ylabel="Distance (mm)", xlim=(0, physical_time(runs[0])[-1]),
                 ylim=(0, max_error * 1.12), xticks=[0, 1, 2], yticks=[0, 5, 10])
    error_ax.text(.98, 1000 * tolerance + .35, "5 mm", color=colors["goal"], fontsize=7,
                  transform=error_ax.get_yaxis_transform(), ha="right")
    for ax in (path_ax, error_ax):
        ax.grid(alpha=.13, lw=.5)
    handles = [Line2D([], [], color=colors["slate"], ls=(0, (3, 2)), lw=.8, label="Reference"),
               Line2D([], [], color=colors["MGATeal"], lw=1.4, label="MGA"),
               Line2D([], [], color=colors["MGAOrange"], lw=1.4, label="w/o LRC")]
    legend = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.60, .035),
                        ncol=3, frameon=True, fancybox=False, edgecolor=colors["divider"],
                        facecolor="white", framealpha=1, fontsize=7.5, borderpad=.4)
    legend.get_frame().set_linewidth(.5)
    save_appendix_surface(fig, output, "surface_execution_geometry", report)


def execution_curves(run):
    """Canonical contact-valid prefix coverage and post-action force impulse.

    The reference is the entire fixed path at every time, not its command prefix.
    Reset states are absent from the canonical coverage signal, so the empty
    prefix at t=0 has zero observed coverage and zero accumulated cost.
    """
    signals = run["signals"]
    positions = np.asarray(signals["coverage_positions"], float)
    reference = np.asarray(signals["reference_path"], float)
    valid = np.asarray(signals["coverage_valid_mask"], float).reshape(-1)
    force = np.asarray(signals["force"], float).reshape(-1)
    env = run["result"]["config_snapshot"]["env_params"]
    dt, force_limit = float(env["dt"]), float(env["f_max"])
    tolerance = float(signals["path_tolerance"])
    if (positions.shape != (100, 3) or reference.shape != (101, 3)
            or valid.shape != (100,) or force.shape != (100,)):
        raise ValueError(f"Execution curves require the complete 100-step schema: {run['path']}")
    if not all(np.isfinite(x).all() for x in (positions, reference, valid, force)):
        raise ValueError(f"Non-finite execution signal: {run['path']}")
    if not (np.isclose(dt, .02) and np.isclose(tolerance, .005)
            and force_limit > 0 and np.isclose(force_limit, signals["f_max"])):
        raise ValueError(f"Unexpected physical clock, tolerance, or force limit: {run['path']}")
    distances = np.linalg.norm(positions[:, None, :] - reference[None, :, :], axis=2)
    hits = (distances <= tolerance) & (valid[:, None] > .5)
    coverage = np.r_[0., 100 * np.logical_or.accumulate(hits, axis=0).mean(axis=1)]
    cost = np.r_[0., dt * np.cumsum(np.maximum(force - force_limit, 0.))]
    # Check representative prefixes with the canonical implementation in
    # addition to record()'s independent stored-terminal-metric check.
    for n in (0, 1, 25, 50, 75, 100):
        canonical = trajectory_path_coverage(positions[:n], reference, valid[:n], tolerance)
        if not np.isclose(coverage[n], 100 * canonical, rtol=0, atol=1e-10):
            raise ValueError(f"Cumulative coverage disagrees with canonical prefix: {run['path']}")
    if not np.isclose(coverage[-1], 100 * run["metrics"]["trajectory_path_coverage"],
                      rtol=0, atol=1e-8):
        raise ValueError(f"Terminal coverage disagrees with saved result: {run['path']}")
    return coverage, cost, dt, force_limit


def plot_surface_execution_curves(root, output, seeds):
    """Opt-in preview only: all paper methods, frozen suites, paired seed bands."""
    colors = paper_colors()
    styles = {method: (title, key, ls) for method, title, key, ls in RESPONSE_METHODS}
    styles.update({"ablation/no_rl_prior": ("w/o RL prior", "MGAIndigo", (0, (6, 2))),
                   "ablation/no_retraction": ("w/o LRC", "MGAOrange", "--")})
    # One shared resampling matrix preserves pairing across methods, outcomes,
    # and physical time. Suites are fixed conditions, not bootstrap units.
    random_seed, replicates = 20260923, 10000
    indices = np.random.default_rng(random_seed).integers(
        0, len(seeds), size=(replicates, len(seeds)))
    data = {
        "scope": "Opt-in saved-execution preview; no existing figure or LaTeX replacement",
        "seeds": list(seeds), "suite_order": list(APPENDIX_SUITES),
        "method_order": [method for method, _ in METHODS],
        "time_seconds": (np.arange(101) * .02).tolist(),
        "run_count": len(METHODS) * len(APPENDIX_SUITES) * len(seeds),
        "definitions": {
            "coverage_percent": "100 * fraction of all 101 reference waypoints visited by any contact-valid saved 3-D point in the observed prefix, distance <= 0.005 m",
            "force_excess_impulse_ns": "C_F(k*dt) = dt * sum_{i=1..k} max(F_n[i] - f_max, 0), in N s; dt=0.02 s",
            "initial_value": "Zero for the empty canonical post-action observation prefix; reset-state contact is not included in the stored coverage metric",
            "clock": "t=0 followed by post-action (step+1)*dt through 2 s; task_signals.seconds is runtime and is not used",
            "aggregation": "Equal mean over all 13 fixed suites within each seed, then equal mean across paired seeds; no smoothing",
            "force_signal": "Recorded force; hybrid is the analytic Winkler reaction, not an independent force sensor",
            "ablation": "no_retraction is the existing no-LRC ablation",
            "uncertainty": "Pointwise percentile bootstrap of seed-level equal-suite means; not a simultaneous confidence band or a suite-population interval",
        },
        "bootstrap": {"confidence": .95, "replicates": replicates, "seed": random_seed,
                      "generator": "numpy.random.default_rng (PCG64)",
                      "unit": "paired seed", "percentiles": [2.5, 97.5],
                      "quantile_interpolation": "linear",
                      "shared_indices_across_methods_metrics_times": True},
        "palette_source": "latex/latex_mga/tex/paper_colors.tex",
        "generator_source": {"path": source_path(Path(__file__)),
                             "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "palette_source_sha256": hashlib.sha256(
            (ROOT / "latex/latex_mga/tex/paper_colors.tex").read_bytes()).hexdigest(),
        "canonical_metric_source": {
            "path": "genedynamics/evaluation/metrics.py",
            "sha256": hashlib.sha256((ROOT / "genedynamics/evaluation/metrics.py").read_bytes()).hexdigest(),
            "function": "trajectory_path_coverage",
            "validated_prefix_lengths": [0, 1, 25, 50, 75, 100]},
        "methods": {}, "sources": [],
    }
    for method, _ in METHODS:
        by_seed = {"coverage_percent": [], "force_excess_impulse_ns": []}
        for seed in seeds:
            suite_coverage, suite_cost = [], []
            for suite in APPENDIX_SUITES:
                run = record(root, method, suite, seed, trajectory=True)
                coverage, cost, dt, force_limit = execution_curves(run)
                suite_coverage.append(coverage); suite_cost.append(cost)
                data["sources"].append({
                    "method": method, "suite": suite, "seed": seed,
                    "results": {"path": source_path(run["path"]),
                                "sha256": hashlib.sha256(run["path"].read_bytes()).hexdigest()},
                    "trajectory": {"path": source_path(run["trajectory_path"]),
                                   "sha256": hashlib.sha256(run["trajectory_path"].read_bytes()).hexdigest()},
                    "dt_seconds": dt, "force_limit_n": force_limit,
                    "reference_waypoints": 101, "post_action_samples": 100,
                    "path_tolerance_m": .005,
                    "terminal_coverage_percent": float(coverage[-1]),
                    "terminal_force_excess_impulse_ns": float(cost[-1]),
                })
            by_seed["coverage_percent"].append(np.mean(suite_coverage, axis=0))
            by_seed["force_excess_impulse_ns"].append(np.mean(suite_cost, axis=0))
        title, key, ls = styles[method]
        summary = {"label": title, "color": colors[key], "line_style": ls,
                   "seed_order": list(seeds), "curves": {}}
        for field, values in by_seed.items():
            values = np.asarray(values)
            resampled_mean = values[indices].mean(axis=1)
            lower, upper = np.percentile(resampled_mean, [2.5, 97.5], axis=0)
            summary["curves"][field] = {
                "per_seed_equal_suite_means": values.tolist(),
                "mean": values.mean(axis=0).tolist(),
                "ci95_lower": lower.tolist(), "ci95_upper": upper.tolist(),
            }
        data["methods"][method] = summary
        print(f"{title}: coverage={summary['curves']['coverage_percent']['mean'][-1]:.6f}%; "
              f"C_F={summary['curves']['force_excess_impulse_ns']['mean'][-1]:.9f} N s", flush=True)

    style()
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8})
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.4))
    fig.subplots_adjust(left=.095, right=.982, top=.79, bottom=.285, wspace=.35)
    fig.text(.095, .962, "Scanning: executed coverage and force-limit cost",
             fontsize=10, fontweight="bold", va="top")
    fig.text(.095, .886, f"13 surface suites | {len(seeds)} paired seeds | mean and 95% bootstrap interval",
             fontsize=8.3, va="top")
    time = np.asarray(data["time_seconds"])
    plot_order = [method for method, _ in METHODS if method != "main/mga"] + ["main/mga"]

    def draw_curves(ax, field, selected, *, inset=False):
        handles = {}
        for method in selected:
            row = data["methods"][method]; curve = row["curves"][field]
            is_mga = method == "main/mga"
            ax.fill_between(time, curve["ci95_lower"], curve["ci95_upper"],
                            color=row["color"], alpha=.15 if is_mga else .075,
                            linewidth=0, zorder=1)
            line, = ax.plot(time, curve["mean"], color=row["color"],
                            ls=styles[method][2], lw=1.7 if is_mga else 1.05,
                            label=row["label"], zorder=4 if is_mga else 3)
            handles[method] = line
        ax.set_xlim(0, 2)
        ax.grid(alpha=.15, lw=.45)
        ax.set_xticks([0, 1, 2] if inset else [0, .5, 1, 1.5, 2])
        return handles

    handles = draw_curves(axes[0], "coverage_percent", plot_order)
    draw_curves(axes[1], "force_excess_impulse_ns", plot_order)
    axes[0].set(xlabel="Physical time (s)", ylabel="Contact-valid coverage (%)",
                ylim=(-2, 103), yticks=[0, 25, 50, 75, 100])
    axes[0].set_title("(a) Visited Reference Path", loc="left", fontweight="bold", pad=7)
    cost_max = max(max(v["curves"]["force_excess_impulse_ns"]["ci95_upper"])
                   for v in data["methods"].values())
    axes[1].set(xlabel="Physical time (s)", ylabel=r"$C_F(t)$ (N s)",
                ylim=(-.025 * cost_max, 1.07 * cost_max))
    axes[1].set_title("(b) Accumulated Force Excess", loc="left", fontweight="bold", pad=7)
    variants = ["ablation/no_rl_prior", "ablation/no_retraction", "main/mga"]
    zoom_max = max(max(data["methods"][m]["curves"]["force_excess_impulse_ns"]["ci95_upper"])
                   for m in variants)
    zoom_top = max(.02, np.ceil(zoom_max * 1.12 / .02) * .02)
    inset = axes[1].inset_axes([.20, .58, .48, .36])
    draw_curves(inset, "force_excess_impulse_ns", variants, inset=True)
    inset.set_ylim(-.025 * zoom_top, zoom_top)
    inset.set_title("MGA Variants (Linear)", fontsize=6.8, loc="left", pad=2)
    inset.tick_params(labelsize=6.5, pad=1, length=1.5)
    inset.set_yticks([0, zoom_top / 2, zoom_top])
    inset.set_yticklabels([f"{value:.2f}" for value in (0, zoom_top / 2, zoom_top)])
    inset.set_facecolor("white")
    for spine in inset.spines.values():
        spine.set_visible(True); spine.set_color(colors["divider"]); spine.set_linewidth(.4)
    legend_order = ["main/mga", "ablation/no_rl_prior", "ablation/no_retraction", "baseline/pegasusflow",
                    "baseline/mppi", "baseline/dial", "baseline/atacom", "baseline/issa"]
    # Matplotlib fills columns: reorder to obtain the intended two legend rows.
    legend = fig.legend(handles=[handles[legend_order[i]] for i in (0, 4, 1, 5, 2, 6, 3, 7)],
                        loc="lower center", bbox_to_anchor=(.54, .025), ncol=4,
                        frameon=True, fancybox=False, edgecolor=colors["divider"],
                        facecolor="white", framealpha=1, fontsize=8,
                        borderpad=.5, columnspacing=1.25, handlelength=2.4, labelspacing=.55)
    legend.get_frame().set_linewidth(.5)
    data["display"] = {"size_inches": [6.8, 3.4], "smoothing": "none",
                       "main_axes": "linear time and linear outcomes",
                       "cost_ylim_ns": list(axes[1].get_ylim()),
                       "cost_inset_methods": variants,
                       "cost_inset_xlim_seconds": [0, 2],
                       "cost_inset_ylim_ns": list(inset.get_ylim()),
                       "inset_axes": "linear; same time and N s units as main cost panel",
                       "legend_row_major": legend_order}
    output.mkdir(parents=True, exist_ok=True)
    save_appendix_surface(fig, output, "surface_coverage_cost", data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "results/arm/surface_scan")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/mga/paper_figures")
    parser.add_argument("--scene-dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--appendix", action="store_true", help="Write additional-result figures only; never replace the main figure")
    parser.add_argument("--appendix-geometry-only", action="store_true", help="With --appendix, only render simulation-linked execution geometry")
    parser.add_argument("--execution-curves", action="store_true", help="Write only the new coverage-cost preview from all saved paper executions")
    args = parser.parse_args()
    if args.execution_curves and (args.appendix or args.appendix_geometry_only):
        parser.error("--execution-curves is a standalone preview mode; do not combine with --appendix")
    if args.appendix_geometry_only and not args.appendix:
        parser.error("--appendix-geometry-only requires --appendix")
    if len(set(args.seeds)) != len(args.seeds) or args.seed not in args.seeds:
        parser.error("Seeds must be unique and include the fixed example seed")
    args.data_root = args.data_root.resolve()
    if args.execution_curves:
        plot_surface_execution_curves(args.data_root, args.output_dir, args.seeds)
        return
    if args.appendix:
        appendix_output = args.output_dir / "appendix"
        if not args.appendix_geometry_only:
            plot_surface_appendix(args.data_root, appendix_output, args.seeds)
        plot_surface_execution_geometry(args.data_root, appendix_output,
                                        args.scene_dir or appendix_output / "scenes", args.seed)
        return
    scene_dir = args.scene_dir or args.output_dir / "scenes"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = [record(args.data_root, method, "hybrid_stripes", args.seed, trajectory=True)
            for method, _ in METHODS]
    scene_run = record(args.data_root, "main/mga", "rigid_bumpy", args.seed, trajectory=True)
    for run in runs[1:]:
        if not np.allclose(run["signals"]["reference_path"], runs[0]["signals"]["reference_path"]):
            raise ValueError("Spatial comparison requires an identical reference path")
    aggregated = {method: summarize(args.data_root, method, "hybrid_stripes", args.seeds)
                  for method, *_ in RESPONSE_METHODS}
    lrc_runs = [record(args.data_root, method, "soft_convex", args.seed, trajectory=True)
                for method in ("main/mga", "ablation/no_retraction")]
    hybrid_runs = [record(args.data_root, method, "hybrid_stripes", args.seed, trajectory=True)
                   for method, *_ in RESPONSE_METHODS]
    lrc_aggregate = {method: summarize(args.data_root, method, "soft_convex", args.seeds)
                     for method in ("main/mga", "ablation/no_retraction")}
    selection = baseline_selection(args.data_root, args.seeds)
    colors = paper_colors()
    style()
    fig = make_figure(runs, scene_run, hybrid_runs, lrc_runs, scene_dir, colors, args.seed)
    for ext in ("pdf", "png"):
        fig.savefig(args.output_dir / f"surface_mechanism.{ext}", dpi=400)
    plt.close(fig)
    sources = []
    for run in runs + hybrid_runs + lrc_runs + [scene_run]:
        for key in ("path", "trajectory_path"):
            path = run[key]
            sources.append({"path": source_path(path),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    report = {
        "example_seed": args.seed, "aggregate_seeds": args.seeds,
        "palette": colors, "source_color_file": "latex/latex_mga/tex/paper_colors.tex",
        "sources": sources, "aggregate": aggregated, "lrc_aggregate": lrc_aggregate,
        "baseline_selection": selection,
        "layout_reference": "reports/mga/scan-peg-60-40-mockup.html",
        "size_inches": [4.2, 3.25], "full_composition_size_inches": [7, 3.25],
        "panel_suites": {"a": "rigid_bumpy", "b": "hybrid_stripes",
                          "c": "hybrid_stripes", "d": "soft_convex"},
        "response_methods": [method for method, *_ in RESPONSE_METHODS],
        "response_display": response_display_report(hybrid_runs, colors),
        "force_color_scale": {"colormap": "mga_manuscript_force_warm", "normalization": "linear",
                              "range": [0, 1], "quantity": "Fn/Fmax",
                              "colors": "90% white + 10% MGAOrange; 30% white + 70% MGAOrange; MGAOrangeInk",
                              "overrange_color": (.8 * np.asarray(to_rgb(colors["MGAOrangeInk"]))).tolist(),
                              "shared_across_methods": True},
        "stiffness_map": {
            "geometry": "plane", "kind": "stripes", "units": "kN/m",
            "nominal_color_limits": [2, 8], "xi_origin": .1, "xi_span": .0495,
            "transition_width": .08, "reference_world_span_mm": 14.85,
            "colormap": "mga_surface_stiffness_cool",
            "colors": "prior_fill; 85% mga_fill + 15% MGATeal",
            "outside_reference_span": "Clipped chart coordinate, not repeated stripes",
            "interpretation": "Nominal reference-chart map, not inferred stiffness at the executed world position",
        },
        "examples": [{"method": method, "coverage_percent": 100 * run["metrics"]["trajectory_path_coverage"]}
                     for (method, _), run in zip(METHODS, runs)],
        "semantics": [
            "Native simulation images replay saved configurations, not freshly simulated dynamics.",
            "Panel (b) vertically projects EE paths onto the reconstructed canonical plane; coverage uses full 3-D saved positions and contact mask.",
            "Panel (b) colors the plane by the actual nominal stiffness-map function. Applied k_surf is evaluated at command scan_xi, not inferred from executed EE world-x.",
            "Material-map limits 2/8 kN/m are nominal parameters; the smooth four-band map attains approximately 2.72/7.28 kN/m.",
            "Hybrid force is the recorded analytic Winkler reaction, not an independent force sensor.",
            "Time is post-action physical time (step+1)*dt; task_signals.seconds is not used.",
            "Panel (c) uses the same hybrid-stripe executions as (b); scan_xi is not executed arc length.",
            "Panel (c) bold traces are a uniform display-only 3-point centered mean; faint traces and triangle peak markers are raw. Smoothing can change peak ordering, so peaks/statistics always use raw values.",
            "No candidate cloud, rejection event, or pre/post-LRC vectors are inferred from execution logs.",
            "w/o LRC reads the existing no_retraction experiment; LRC is not a physical retract action.",
            "Panel (a) shows rigid_bumpy; (b)/(c) show hybrid_stripes; (d) shows soft_convex, all with the same example seed.",
            "Panel (d) is soft_convex, same fixed example seed; error is canonical 3-D nearest-polyline distance, not h_surf or a safety violation.",
        ],
    }
    (args.output_dir / "surface_mechanism_data.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(args.output_dir / "surface_mechanism.pdf")


if __name__ == "__main__":
    main()
