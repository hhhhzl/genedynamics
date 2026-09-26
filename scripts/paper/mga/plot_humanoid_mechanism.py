"""Plot the H1 mechanism figure from canonical executed records and scene renders.

No environment is stepped and no result file is modified. Scene images must be
rendered from the same recorded seed; the companion JSON records every input.
Native task evaluation trimming excludes absorbing success padding and retains
the first terminal physical transition. Rejected actions are never plotted as
executed states. Run from the repository environment, for example::

    python scripts/paper/mga/plot_humanoid_mechanism.py --require-scenes
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.path import Path as PlotPath
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FORCE_METHODS = (
    ("main/mga", "MGA", "teal", "-"),
    ("baseline/mppi", "MPPI", "goal_border", "--"),
    ("baseline/dial", "DIAL", "orange", ":"),
    ("baseline/pegasusflow", "PegasusFlow", "baseline_blue", "-."),
    ("baseline/atacom", "ATACOM", "indigo", (0, (4, 1.5, 1, 1.5))),
    ("baseline/issa", "ISSA", "slate", (0, (2.5, 1))),
)
PHASE_METHODS = (
    ("main/mga", "MGA", "teal", "-"),
    ("ablation/no_retraction", "w/o LRC", "orange", "--"),
)
SCENES = {
    "force": "humanoid_force_regulation.png",
    "push": "humanoid_fixed_stance_push.png",
    "unjam": "humanoid_unjamming.png",
}


@dataclass
class Run:
    method: str
    suite: str
    seed: int
    result_path: Path
    trajectory_path: Path
    result: dict
    signals: dict
    execution_status: dict

    @property
    def metrics(self):
        return self.result["metrics"]["humanoid_box_push_metrics"]


def parse_seeds(text):
    """Accept inclusive ranges or comma-separated seeds, rejecting duplicates."""
    seeds = []
    for part in text.split(","):
        if re.fullmatch(r"\d+-\d+", part.strip()):
            start, end = map(int, part.split("-"))
            if end < start:
                raise argparse.ArgumentTypeError("Seed ranges must be increasing")
            seeds.extend(range(start, end + 1))
        elif part.strip().isdigit():
            seeds.append(int(part))
        else:
            raise argparse.ArgumentTypeError("Use --seeds 0-9 or --seeds 0,1,2")
    if len(set(seeds)) != len(seeds) or not seeds:
        raise argparse.ArgumentTypeError("Seeds must be nonempty and unique")
    return sorted(seeds)


def palette():
    path = ROOT / "latex/latex_mga/tex/paper_colors.tex"
    definitions = dict(re.findall(
        r"\\definecolor\{([^}]+)\}\{HTML\}\{([0-9a-fA-F]{6})\}",
        path.read_text(),
    ))
    colors = {name: "#" + definitions[tex_name] for name, tex_name in (
        ("teal", "MGATeal"), ("indigo", "MGAIndigo"),
        ("orange", "MGAOrange"), ("orange_ink", "MGAOrangeInk"),
        ("baseline_blue", "MGAAppendixInk"),
    )}
    # User-specified stage fills complement the established paper accents.
    # Keep these local: a preview must not rewrite the paper's color macros.
    colors.update({
        "prior_fill": "#E3DDEF", "mga_fill": "#D6EDE8",
        "deployment_fill": "#E4F1D3", "goal_fill": "#EDF3C9",
        "goal_border": "#5C8F2E", "slate": "#3A4655",
        "gray": "#9A9A9A",
    })
    return colors, path


def load_run(data_root, method, suite, seed):
    from genedynamics.experiments.plugins.metrics.extractors import _humanoid_evaluation_view

    directory = data_root / method / f"level_{suite}" / f"seed_{seed}"
    result_path = directory / "results.json"
    trajectory_path = directory / "trajectory/trajectory.json"
    result = json.loads(result_path.read_text())
    trajectory = json.loads(trajectory_path.read_text())
    if int(result["seed"]) != seed:
        raise ValueError(f"Seed mismatch: {result_path}")
    # Reuse task-owned execution semantics rather than introducing a second
    # metrics protocol in a plotting script.
    signals = _humanoid_evaluation_view(trajectory["task_signals"])
    if not np.all(np.asarray(signals["physics_samples_valid"], bool)):
        raise ValueError(f"Incomplete executed physics evidence: {trajectory_path}")
    return Run(method, suite, seed, result_path, trajectory_path, result,
               signals, trajectory.get("execution_status", result.get("execution_status", {})))


def force_trace(run):
    force = np.asarray(run.signals["physics_hand_force"], float)
    dt = float(run.signals["dt"])
    physics_dt = float(run.signals["physics_dt"])
    # Native clock: control_start + (substep + 1) * physics_dt.
    times = np.arange(force.shape[0])[:, None] * dt + (
        np.arange(force.shape[1])[None, :] + 1
    ) * physics_dt
    return times.reshape(-1), force.reshape(-1)


def phase_trace(run):
    signals = run.signals
    target = float(np.asarray(signals["target"]).reshape(-1)[0])
    target_yaw = float(np.asarray(signals["target_yaw"]).reshape(-1)[0])
    remaining = np.abs(target - np.asarray(signals["box_x"], float)) * 1000
    yaw_delta = np.asarray(signals["box_yaw"], float) - target_yaw
    yaw = np.rad2deg(np.abs(np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta))))
    return remaining, yaw


def stats(values):
    values = np.asarray(values, float)
    return {"values": values.tolist(), "mean": float(values.mean()),
            "std": float(values.std(ddof=0)), "std_ddof": 0, "n": len(values)}


def source_path(path):
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def validate_scenes(groups, scene_dir, seed, required):
    metadata_path = scene_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    verified = {}
    for key, suite in (("force", "p1_force_15n"), ("push", "p2_push_nominal"),
                       ("unjam", "p3_unjam")):
        path = scene_dir / SCENES[key]
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Render the recorded scene first: {path}")
            continue
        run = next(r for r in groups[("main/mga", suite)] if r.seed == seed)
        record = metadata.get(path.stem, {})
        if record.get("seed") != seed:
            raise ValueError(f"Scene seed metadata is missing or stale; re-render {path}")
        if record.get("capture_pending", False):
            raise ValueError(f"Scene browser capture is still pending: {path}")
        if record.get("state_indices") != [0, record.get("last_unpadded_state")]:
            raise ValueError(f"Scene must show only the recorded first and final unpadded states: {path}")
        if (record.get("image_sha256") is not None
                and hashlib.sha256(path.read_bytes()).hexdigest() != record["image_sha256"]):
            raise ValueError(f"Scene image hash is stale; finish rendering {path}")
        for name, source in (("trajectory", run.trajectory_path), ("result", run.result_path)):
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if record.get(name + "_sha256") != digest:
                raise ValueError(f"Scene {name} hash is missing or stale; re-render {path}")
        camera = record.get("camera", {})
        elevation = camera.get("elevation", float("nan"))
        # Native Brax Web records an eye/target camera, whereas the MuJoCo
        # preview records azimuth/elevation. Both describe the actual view.
        if "position" in camera and "target" in camera:
            eye = np.asarray(camera["position"], float)
            target = np.asarray(camera["target"], float)
            if eye.shape != (3,) or target.shape != (3,) or not np.all(np.isfinite([eye, target])):
                raise ValueError(f"Invalid recorded camera vectors: {path}")
            direction = target - eye
            elevation = np.rad2deg(np.arctan2(direction[2], np.linalg.norm(direction[:2])))
        valid_view = (np.isclose(elevation, -90.0) if key == "unjam"
                      else np.isfinite(elevation) and -50.0 <= elevation <= 10.0)
        if not valid_view:
            raise ValueError(f"Scene needs the requested {'top' if key == 'unjam' else 'side/oblique'} view; re-render {path}")
        verified[key] = record
    return verified


def style(colors):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans", "font.size": 7,
        "axes.labelsize": 7, "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.linewidth": 0.55, "xtick.major.width": 0.55,
        "ytick.major.width": 0.55, "xtick.major.size": 2,
        "ytick.major.size": 2, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42,
        "ps.fonttype": 42, "savefig.facecolor": "white",
        "text.color": colors["slate"], "axes.labelcolor": colors["slate"],
        "axes.edgecolor": colors["slate"], "xtick.color": colors["slate"],
        "ytick.color": colors["slate"],
    })


def fixed_axes(fig, bounds):
    """Create an axes from physical inches; never let image aspect resize it."""
    x, y, width, height = bounds
    return fig.add_axes([x / fig.get_figwidth(), y / fig.get_figheight(),
                         width / fig.get_figwidth(), height / fig.get_figheight()])


def scene(ax, path, required, contain=False):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Render the recorded scene first: {path}")
        ax.text(0.5, 0.5, "Scene render pending", transform=ax.transAxes,
                ha="center", va="center", color="0.45")
        return
    pixels = plt.imread(path)
    pixel_height, pixel_width = pixels.shape[:2]
    box = ax.get_position(original=True)
    width = box.width * ax.figure.get_figwidth()
    height = box.height * ax.figure.get_figheight()
    # Isotropic contain/cover in physical coordinates: model proportions are
    # unchanged and the exact panel rectangle is never shrunk by imshow.
    scale = (min if contain else max)(width / pixel_width, height / pixel_height)
    draw_width, draw_height = pixel_width * scale, pixel_height * scale
    x0, y0 = (width - draw_width) / 2, (height - draw_height) / 2
    ax.set_facecolor(np.mean(pixels[:4, :4, :3], axis=(0, 1)))
    ax.imshow(pixels, extent=(x0, x0 + draw_width, y0, y0 + draw_height),
              aspect="auto", interpolation="lanczos")
    ax.set(xlim=(0, width), ylim=(0, height))


def panel_frame(fig, bounds, letter, title, colors):
    x, y, width, height = bounds
    fig.add_artist(Rectangle((x / fig.get_figwidth(), y / fig.get_figheight()),
                            width / fig.get_figwidth(), height / fig.get_figheight(),
                            transform=fig.transFigure, facecolor="none",
                            edgecolor=colors["gray"], linewidth=0.45, zorder=30))
    title_y = (y + height + 0.035) / fig.get_figheight()
    fig.text(x / fig.get_figwidth(), title_y, f"({letter})",
             color=colors["orange_ink"], weight="bold", ha="left", va="bottom")
    fig.text((x + 0.175) / fig.get_figwidth(), title_y, title,
             color=colors["slate"], weight="bold", ha="left", va="bottom")


def scene_direction_arrows(axes, scene_dir, scene_metadata, colors):
    """Project direction-only vector annotations onto verified saved scenes.

    The force cue is the commanded rear-face push convention, not a measured
    three-axis force vector. Translation and yaw signs come from native first
    and final box poses. Annotation lengths/arc spans never encode magnitude.
    """
    evidence = {}
    for key, ax in axes.items():
        record = (scene_metadata or {}).get(key)
        if not record:
            continue
        payload_path = scene_dir / record["web_payload"]
        html_path = scene_dir / record["web_html"]
        for path, field in ((payload_path, "web_payload_sha256"),
                            (html_path, "web_html_sha256")):
            if hashlib.sha256(path.read_bytes()).hexdigest() != record[field]:
                raise ValueError(f"Stale camera/pose evidence for scene arrows: {path}")
        paper = json.loads(payload_path.read_text())["paper"]
        fov_match = re.search(r"PerspectiveCamera\(([\d.]+),p\.width/p\.height", html_path.read_text())
        if fov_match is None:
            raise ValueError(f"Cannot verify native camera field of view: {html_path}")
        fov = float(fov_match.group(1))
        if paper["pose_indices"] != record["state_indices"] or paper["camera"] != record["camera"]:
            raise ValueError(f"Scene arrows and displayed poses use different camera/states: {payload_path}")
        eye = np.asarray(paper["camera"]["position"], float)
        forward = np.asarray(paper["camera"]["target"], float) - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.asarray(paper["camera"]["up"], float))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        width, height = float(np.diff(ax.get_xlim())[0]), float(np.diff(ax.get_ylim())[0])
        image_width, image_height = paper["width"], paper["height"]
        scale = max(width / image_width, height / image_height)
        draw = np.asarray([image_width, image_height]) * scale
        offset = (np.asarray([width, height]) - draw) / 2

        def project(points):
            rel = np.asarray(points, float) - eye
            depth = rel @ forward
            if np.any(depth <= 0):
                raise ValueError("Direction annotation lies behind the native camera")
            denominator = depth * np.tan(np.deg2rad(fov / 2))
            uv = np.stack(((rel @ right) / denominator / (image_width / image_height),
                           (rel @ up) / denominator), axis=-1) * 0.5 + 0.5
            return offset + uv * draw

        def arrow(start, end, *, dashed=False, points=None):
            kwargs = ({"path": PlotPath(points)} if points is not None
                      else {"posA": start, "posB": end, "shrinkA": 0, "shrinkB": 0})
            patch = FancyArrowPatch(**kwargs, arrowstyle="-|>", mutation_scale=7.0,
                                    linewidth=1.15, linestyle=(0, (2.6, 1.5)) if dashed else "-",
                                    color=colors["orange"], zorder=16)
            patch.set_path_effects([path_effects.Stroke(linewidth=1.95, foreground="white"),
                                   path_effects.Normal()])
            ax.add_patch(patch)

        def label(text, point):
            artist = ax.text(*point, text, fontsize=6.0, ha="center", va="center",
                             color=colors["orange"], zorder=17)
            artist.set_path_effects([path_effects.Stroke(linewidth=1.25, foreground="white"),
                                    path_effects.Normal()])

        first, final = paper["box_poses"][0], paper["box_poses"][-1]
        p0, p1 = np.asarray(first["position"]), np.asarray(final["position"])
        r0, r1 = np.asarray(first["rotation"]), np.asarray(final["rotation"])
        half = np.asarray(paper["box_half_size"])
        displacement = p1 - p0
        yaw0, yaw1 = np.arctan2(r0[1, 0], r0[0, 0]), np.arctan2(r1[1, 0], r1[0, 0])
        delta_yaw = float(np.arctan2(np.sin(yaw1 - yaw0), np.cos(yaw1 - yaw0)))
        item = {"payload": source_path(payload_path), "payload_sha256": record["web_payload_sha256"],
                "camera_fov_y_degrees": fov, "state_indices": paper["pose_indices"],
                "box_displacement_m": displacement.tolist(),
                "box_yaw_first_last_degrees": np.rad2deg([yaw0, yaw1]).tolist(),
                "box_yaw_delta_degrees": float(np.rad2deg(delta_yaw)),
                "encoding": "Direction only: displayed arrow length/arc span is not measured magnitude"}
        if key == "force":
            # _unpack fixes the rear face for push_to_line, _contact_target
            # uses outward normal R[-1,0,0], and hand_contact_wrench applies
            # -desired_force*normal. Thus the commanded push direction is R[:,0].
            direction = r1[:, 0]
            anchor = p1 + r1 @ np.array([-half[0], -0.20, 0.42])
            start = project(anchor)
            screen_direction = project(anchor + direction) - start
            screen_direction /= np.linalg.norm(screen_direction)
            end = start + 0.28 * screen_direction
            arrow(start, end, dashed=True)
            label("$F$", (start + end) / 2 + [0, 0.06])
            item.update(kind="commanded rear-face force direction", line_style="dashed",
                        world_direction=direction.tolist(), display_length_inches=0.28,
                        source_convention=["genedynamics/envs/domains/humanoid/box_push_brax.py:_unpack/_contact_target",
                                           "genedynamics/core/control/humanoid_contact.py:hand_contact_wrench"])
        else:
            if np.linalg.norm(displacement[:2]) <= 1e-8:
                raise ValueError(f"No nonzero actual box translation for a motion arrow: {key}")
            anchor = p1 + ([0, -half[1], 0.10] if key == "push"
                           else [0, -0.20, half[2] + 0.015])
            center = project(anchor)
            screen_direction = project(anchor + displacement) - center
            screen_direction /= np.linalg.norm(screen_direction)
            start, end = center - 0.21 * screen_direction, center + 0.21 * screen_direction
            arrow(start, end)
            label("$\\Delta x$", center + [0, -0.065])
            item.update(kind="actual first-to-final box displacement direction", line_style="solid",
                        world_direction=(displacement / np.linalg.norm(displacement)).tolist(),
                        display_length_inches=0.42)
            if key == "unjam" and abs(delta_yaw) > 1e-8:
                center_world = p1 + [0, 0.10, half[2] + 0.02]
                angles = np.deg2rad(np.linspace(140, 140 + np.sign(delta_yaw) * 100, 45))
                arc_world = center_world + np.column_stack((0.30 * np.cos(angles),
                                                           0.30 * np.sin(angles), np.zeros_like(angles)))
                arc = project(arc_world)
                arrow(arc[0], arc[-1], points=arc)
                label("$\\Delta\\psi$", project(center_world) + [0.01, 0.03])
                item["yaw_cue"] = {"world_rotation": "clockwise about +z" if delta_yaw < 0 else "counterclockwise about +z",
                                   "display_arc_span_degrees": 100,
                                   "meaning": "Orientation-correction direction, not the executed rotation magnitude"}
        evidence[key] = item
    return evidence


def create_figure(groups, scene_dir, seed, colors, require_scenes, scene_metadata=None):
    # Widen the equal 2x2 frames: a square is not a meaningful constraint on
    # a scientific plot. A retains an independent, readable force-time axes
    # below its native scene rather than overlaying a miniature inset.
    fig = plt.figure(figsize=(4.00, 3.45))
    cells = [(0.14, 1.97, 1.77, 1.31), (2.10, 1.97, 1.77, 1.31),
             (0.14, 0.45, 1.77, 1.31), (2.10, 0.45, 1.77, 1.31)]
    ax, ay, aw, ah = cells[0]
    force_scene_bounds = (ax, ay + 0.79, aw, ah - 0.79)
    force_plot_bounds = (ax + 0.29, ay + 0.24, aw - 0.34, 0.52)
    force_scene = fixed_axes(fig, force_scene_bounds)
    force_ax = fixed_axes(fig, force_plot_bounds)
    push_ax = fixed_axes(fig, cells[1])
    unjam_ax = fixed_axes(fig, cells[2])
    dx, dy, dw, dh = cells[3]
    phase_ax = fixed_axes(fig, (dx + 0.265, dy + 0.235, dw - 0.31, dh - 0.275))

    scene(force_scene, scene_dir / SCENES["force"], require_scenes)
    scene(push_ax, scene_dir / SCENES["push"], require_scenes)
    scene(unjam_ax, scene_dir / SCENES["unjam"], require_scenes)
    for key, scene_ax in (("force", force_scene), ("push", push_ax), ("unjam", unjam_ax)):
        record = (scene_metadata or {}).get(key)
        if record is None:
            continue
        times = np.asarray(record["time_seconds"], float)
        if times.ndim != 1 or not len(times) or not np.all(np.isfinite(times)):
            raise ValueError(f"Invalid scene timestamp metadata: {key}")
        # This is the actual ghost-frame range, not a fabricated animation
        # duration or the force-controller/planner wall clock.
        begin, end = (f"{float(t):.2f}".rstrip("0").rstrip(".")
                      for t in (times.min(), times.max()))
        time_label = (f"poses: {begin}→{end} s" if key == "unjam"
                      else f"t={begin}→{end} s")
        scene_ax.text(0.02, 0.018, time_label, transform=scene_ax.transAxes,
                       ha="left", va="bottom", fontsize=5.5, color=colors["slate"],
                       bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9,
                             "pad": 0.7}, zorder=10)
    for key, scene_ax in (("push", push_ax), ("unjam", unjam_ax)):
        record = (scene_metadata or {}).get(key, {})
        outline_meta = record.get("box_motion_annotation")
        if not outline_meta:
            if require_scenes:
                raise ValueError(f"Scene must record its actual initial/final box outlines: {key}")
            continue
        # The two native, unshifted box outlines use the same stage colors as
        # the robot silhouettes. Read those actual colors rather than applying
        # an independent figure key that could disagree with the rendering.
        outline_colors = outline_meta.get("outline_colors")
        if outline_colors is None:
            if str(record.get("render_style", "")).startswith("brax-web"):
                raise ValueError("Brax scene must record its actual initial/final outline colors")
            outline_colors = {"initial": "#F2FAFF", "final": colors["orange"]}
        scene_ax.legend(handles=[
            Line2D([], [], color=outline_colors["initial"], lw=1.5, label="Initial"),
            Line2D([], [], color=outline_colors["final"], lw=1.5, label="Final"),
        ], loc="upper right", bbox_to_anchor=(0.98, 0.98), ncol=2,
            frameon=True, facecolor="white", edgecolor=colors["gray"], framealpha=0.94,
            labelcolor=colors["slate"], fontsize=6.3, handlelength=1.2,
            handletextpad=0.35, columnspacing=0.85, borderpad=0.3, borderaxespad=0)
    for bounds, letter, title in zip(cells, "abcd",
                                     ("Force control", "Fixed-stance", "Unjamming", "Continuation")):
        panel_frame(fig, bounds, letter, title, colors)
    arrow_evidence = scene_direction_arrows(
        {"force": force_scene, "push": push_ax, "unjam": unjam_ax},
        scene_dir, scene_metadata, colors)

    force_handles = []
    force_peak = 0.0
    force_end = 0.0
    plotted_peaks = {}
    for method, label, color, line_style in FORCE_METHODS:
        run = next(r for r in groups[(method, "p1_force_15n")] if r.seed == seed)
        time, force = force_trace(run)
        if not len(force) or not np.all(np.isfinite(force)):
            raise ValueError(f"Missing or nonfinite executed force samples: {run.trajectory_path}")
        force_peak = max(force_peak, float(force.max()))
        force_end = max(force_end, float(time[-1]))
        peak_index = int(np.argmax(force))
        plotted_peaks[label] = {"force_N": float(force[peak_index]), "time_s": float(time[peak_index])}
        line, = force_ax.plot(time, force, color=colors[color], ls=line_style,
                              lw=1.35 if method == "main/mga" else 0.7,
                              alpha=1.0 if method == "main/mga" else 0.82,
                              zorder=4 if method == "main/mga" else 2,
                              label=label)
        if method == "main/mga":
            line.set_path_effects([path_effects.Stroke(linewidth=2.05, foreground="white"),
                                   path_effects.Normal()])
        force_ax.plot(time[peak_index], force[peak_index], "o", color=colors[color],
                      ms=2.7 if method == "main/mga" else 2.0,
                      mec="white", mew=0.35, zorder=6)
        force_handles.append(line)
    reference_run = next(r for r in groups[("main/mga", "p1_force_15n")] if r.seed == seed)
    reference_time, _ = force_trace(reference_run)
    n_substeps = np.asarray(reference_run.signals["physics_hand_force"]).shape[1]
    reference = np.repeat(np.asarray(reference_run.signals["task_force_reference"]), n_substeps)
    for method, *_ in FORCE_METHODS:
        other = next(r for r in groups[(method, "p1_force_15n")] if r.seed == seed)
        other_time, _ = force_trace(other)
        other_ref = np.repeat(np.asarray(other.signals["task_force_reference"]),
                              np.asarray(other.signals["physics_hand_force"]).shape[1])
        if not (np.array_equal(reference_time, other_time) and np.array_equal(reference, other_ref)):
            raise ValueError("Force references differ across methods; a shared reference line would be ambiguous")
    force_ax.step(reference_time, reference, color=colors["slate"], lw=0.6,
                  ls=(0, (2, 2)), where="post", zorder=1)
    # Include every observed peak, including seeds with larger impacts; no
    # fixed y-limit may hide a new baseline's force transient.
    force_top = float(np.ceil(max(force_peak, float(reference.max())) * 1.08 / 10) * 10)
    force_ax.text(force_end * 0.98, float(reference[-1]) + force_top * 0.06,
                  f"{float(reference[-1]):g} N", color=colors["slate"],
                  ha="right", fontsize=6.0)
    force_ax.set(xlim=(0, force_end), ylim=(0, force_top),
                 xticks=[0, force_end / 2, force_end])
    force_ax.locator_params(axis="y", nbins=3)
    force_ax.set_xlabel("Time (s)", fontsize=6.5, labelpad=0.8)
    force_ax.set_ylabel("$F$ (N)", fontsize=6.5, labelpad=1.2)
    force_ax.tick_params(labelsize=6.0, pad=0.8, length=1.5)
    force_ax.grid(axis="y", color=colors["gray"], alpha=0.25, lw=0.45)
    other_peaks = [value["force_N"] for label, value in plotted_peaks.items() if label != "MGA"]
    for y, text, color in (
            (0.94, f"MGA peak {plotted_peaks['MGA']['force_N']:.1f} N", "teal"),
            (0.71, f"Others {min(other_peaks):.1f}–{max(other_peaks):.1f} N", "slate")):
        force_ax.text(0.98, y, text, transform=force_ax.transAxes, ha="right", va="top",
                      fontsize=5.8, color=colors[color],
                      bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 0.3})

    example = groups[("main/mga", "p3_unjam")][0]
    params = example.result["config_snapshot"]["env_params"]
    distance_tol = float(example.signals["goal_tolerance"]) * 1000
    yaw_tol = np.rad2deg(float(params["unjam_yaw_eps"]))
    phase_ax.add_patch(Rectangle((0, 0), distance_tol, yaw_tol,
                                facecolor=colors["goal_fill"], edgecolor="none"))
    phase_ax.plot([0, distance_tol, distance_tol], [yaw_tol, yaw_tol, 0],
                  color=colors["goal_border"], lw=0.65)
    max_yaw = 0.0
    for method, _, color, line_style in PHASE_METHODS:
        for run in groups[(method, "p3_unjam")]:
            remaining, yaw = phase_trace(run)
            max_yaw = max(max_yaw, float(yaw.max()))
            bold = run.seed == seed
            phase_ax.plot(remaining, yaw, color=colors[color], ls=line_style,
                          alpha=1 if bold else 0.19, lw=0.95 if bold else 0.45,
                          zorder=4 if bold else 2)
            stopped = run.execution_status.get("reason") == "no_revalidated_safe_candidate"
            phase_ax.scatter(remaining[-1], yaw[-1], color=colors[color],
                             marker="x" if stopped else "o", s=9 if bold else 5,
                             linewidths=0.8 if stopped else 0.2,
                             alpha=1 if bold else 0.55, zorder=5)
    phase_ax.set(xlim=(-1, 32), ylim=(-0.12, max(5.0, max_yaw * 1.12)),
                 xticks=[0, 15, 30], yticks=[0, 2, 4])
    phase_ax.set_xlabel("Goal error (mm)", fontsize=6, labelpad=1.2)
    phase_ax.set_ylabel("Yaw error (deg)", fontsize=6, labelpad=1.4)
    phase_ax.tick_params(labelsize=5.5, pad=1.0, length=1.5)
    phase_ax.grid(color=colors["gray"], alpha=0.25, lw=0.45)
    phase_ax.annotate("Goal", xy=(2.5, 0.2), xytext=(9, 0.3),
                      color=colors["goal_border"], fontsize=5.8,
                      arrowprops={"arrowstyle": "-", "color": colors["goal_border"], "lw": 0.5})
    phase_ax.legend(handles=[
        Line2D([], [], color=colors["teal"], lw=1.1, label="MGA"),
        Line2D([], [], color=colors["orange"], lw=1.1, ls="--", label="w/o LRC"),
    ], loc="upper left", bbox_to_anchor=(-0.015, 1.02), frameon=False,
        fontsize=5.8, handlelength=0.9, handletextpad=0.35,
        labelspacing=0.2, borderpad=0.1)

    # Matplotlib fills legend columns first; reorder for the method sequence
    # MGA / MPPI / DIAL on the first row and the other baselines below.
    legend_handles = [force_handles[i] for i in (0, 3, 1, 4, 2, 5)]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.50, 0.012),
               ncol=3, frameon=True, facecolor="white", edgecolor=colors["gray"],
               fontsize=7, handlelength=1.9, labelspacing=0.3,
               handletextpad=0.4, columnspacing=1.3, borderpad=0.4,
               borderaxespad=0)
    phase_ax.text(0.98, 0.92, "× stop", transform=phase_ax.transAxes,
                  ha="right", color=colors["orange"], fontsize=5.8)
    fig._mga_panel_bounds_inches = cells
    fig._mga_force_plot_bounds_inches = force_plot_bounds
    fig._mga_force_scene_bounds_inches = force_scene_bounds
    fig._mga_arrow_evidence = arrow_evidence
    fig._mga_force_highlight = {"raw_peaks_by_method": plotted_peaks,
                                "semantics": "Unsmoothed 4 ms force traces; dots mark actual maxima; callout compares only the six displayed methods at the common seed",
                                "mga_style": "1.35 pt raw trace with slim white halo; no smoothing or impact removal"}
    return fig


APPENDIX_CONTINUATION_METHODS = (
    ("main/mga", "MGA", "teal"),
    ("ablation/no_rl_prior", "w/o RL prior", "indigo"),
    ("ablation/no_retraction", "w/o LRC", "orange"),
)


def appendix_run_source(run):
    return {"method": run.method, "suite": run.suite, "seed": run.seed,
            "result": source_path(run.result_path),
            "result_sha256": hashlib.sha256(run.result_path.read_bytes()).hexdigest(),
            "trajectory": source_path(run.trajectory_path),
            "trajectory_sha256": hashlib.sha256(run.trajectory_path.read_bytes()).hexdigest()}


def save_humanoid_appendix(fig, output, name, data):
    stem = output / name
    for extension in ("pdf", "png"):
        fig.savefig(stem.with_suffix("." + extension), dpi=300, facecolor="white")
    plt.close(fig)
    stem.with_name(stem.name + "_data").with_suffix(".json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n")
    print(stem.with_suffix(".pdf"))


def safe_completion_record(run):
    """Verify a completed trial before assigning its actual completion clock.

    A rejected execution may have a physically safe observed prefix, but that
    prefix never establishes a full-trial success or an unobserved trajectory.
    ``load_run`` has already applied the task-owned initial/substep safety and
    terminal-padding protocol; do not replace it with endpoint force checks.
    """
    signals = run.signals
    dt = float(signals["dt"])
    requested_steps = int(run.result["config_snapshot"]["n_steps"])
    retained_steps = len(signals["task_success"])
    events = np.flatnonzero(np.asarray(signals["task_success"]) > .5)
    first_success = (int(events[0]) + 1) * dt if events.size else None
    violation = float(signals["safe_success_violation"])
    coverage_complete = bool(np.all(signals["physics_samples_valid"]))
    safety_valid = coverage_complete and np.isfinite(violation) and violation <= 0
    aborted = run.execution_status.get("state") == "aborted_unrecoverable"
    canonical_ssr = float(run.metrics["safe_success"])
    if canonical_ssr not in (0.0, 1.0):
        raise ValueError(f"Nonbinary canonical safe_success: {run.result_path}")
    goal_valid = float(signals["safe_success_goal_error"]) <= float(signals["goal_tolerance"])
    verified_success = bool(events.size and safety_valid and goal_valid and not aborted)
    if verified_success != bool(canonical_ssr):
        raise ValueError(f"Canonical SSR disagrees with executed safety/completion evidence: {run.result_path}")
    if first_success is not None and first_success > requested_steps * dt + 1e-9:
        raise ValueError(f"Completion falls outside the evaluation horizon: {run.trajectory_path}")
    abort_time = None
    if aborted:
        executed = int(run.execution_status["executed_steps"])
        if executed != retained_steps or int(run.execution_status["rejected_step"]) != executed:
            raise ValueError(f"Abort time does not match the retained execution prefix: {run.trajectory_path}")
        abort_time = executed * dt
    return {
        **appendix_run_source(run),
        "outcome": ("safe_completion" if verified_success else "aborted_unrecoverable" if aborted
                    else "unsafe_execution" if not safety_valid else "incomplete"),
        "canonical_safe_success": int(canonical_ssr),
        "safe_completion_verified": verified_success,
        "first_recorded_task_success_time_s": first_success,
        "safe_completion_time_s": first_success if verified_success else None,
        "abort_time_s": abort_time,
        "abort_reason": run.execution_status.get("reason") if aborted else None,
        "retained_actual_control_steps": retained_steps,
        "actual_physics_end_time_s": retained_steps * dt,
        "requested_steps": requested_steps,
        "horizon_s": requested_steps * dt,
        "dt_s": dt,
        "physics_dt_s": float(signals["physics_dt"]),
        "physics_coverage_complete": coverage_complete,
        "safety_valid_over_observed_initial_and_active_window": bool(safety_valid),
        "physics_max_violation": violation,
        "safety_scope": "observed_execution_prefix_only" if aborted else "complete_task_evaluation_window",
        "initial_safety_margins": np.asarray(signals["physics_initial_safety_margins"]).tolist(),
        "evaluation_contract": signals["task_metadata"]["evaluation_contract"],
        "recorded_source_sha256": signals["task_metadata"]["reliability_source_sha256"],
    }


def plot_humanoid_safe_completion(root, output, seeds):
    """Empirical safe-completion CDF; failed attempts keep their denominator."""
    if seeds != list(range(10)):
        raise ValueError("The controlled completion preview requires all paired seeds 0-9")
    colors, color_path = palette()
    style(colors)
    plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5,
                         "xtick.labelsize": 8, "ytick.labelsize": 8})
    source_files = (Path(__file__), color_path,
                    ROOT / "genedynamics/experiments/plugins/metrics/extractors.py")
    report = {
        "figure": "Safe completion during unjamming",
        "suite": "p3_unjam", "seeds": seeds, "trials_per_method": len(seeds),
        "curve_definition": "F_m(t) = sum_i 1[verified safe completion in trial i at time <= t] / 10",
        "time_definition": "Physical execution time: (zero-based post-transition success row + 1) * recorded control dt",
        "failure_rule": "Aborted, unsafe and incomplete attempts never contribute a completion event and remain in the fixed denominator at every deadline",
        "padding_rule": "Task-owned extractor removes unexecuted success padding; after success only the completion indicator persists, not force/state samples",
        "uncertainty": "Empirical fractions only; no confidence bands. A 10/10 sample fraction is not a population-success guarantee.",
        "method_definitions": {
            "main/mga": "Full MGA",
            "ablation/no_rl_prior": "MGA with learned RL proposal prior removed",
            "ablation/no_retraction": "MGA with local residual correction (LRC) disabled; tangent shaping and remaining components retained",
        },
        "scope": "Controlled MGA ablations, not an all-baseline comparison or an RL training curve",
        "generator_sources": [{"path": source_path(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                              for p in source_files],
        "methods": {}, "sources": [],
    }
    fig = plt.figure(figsize=(5.5, 3.3))
    ax = fig.add_axes([.12, .32, .77, .48])
    handles = []
    common_dt = common_horizon = None
    for (method, label_text, key), ls in zip(APPENDIX_CONTINUATION_METHODS, ("-", "--", "-.")):
        rows = [safe_completion_record(load_run(root, method, "p3_unjam", seed)) for seed in seeds]
        for row in rows:
            if common_dt is None:
                common_dt, common_horizon = row["dt_s"], row["horizon_s"]
            if not np.isclose(row["dt_s"], common_dt) or not np.isclose(row["horizon_s"], common_horizon):
                raise ValueError("Completion curves require matched physical clocks and evaluation horizons")
        grid = np.arange(int(round(common_horizon / common_dt)) + 1) * common_dt
        completed_times = [r["safe_completion_time_s"] for r in rows if r["safe_completion_verified"]]
        counts = np.asarray([sum(t <= deadline + 1e-9 for t in completed_times) for deadline in grid])
        fraction = counts / len(rows)
        ax.step(grid, 100 * fraction, where="post", color=colors[key], ls=ls,
                lw=1.9 if method == "main/mga" else 1.65, zorder=4 if method == "main/mga" else 3)
        ax.scatter([common_horizon], [100 * fraction[-1]], s=15, color=colors[key],
                   edgecolors="white", linewidths=.35, zorder=5, clip_on=False)
        ax.text(common_horizon + .045, 100 * fraction[-1], f"{counts[-1]}/{len(rows)}",
                color=colors[key], fontsize=9, fontweight="bold", va="center", clip_on=False)
        handles.append(Line2D([], [], color=colors[key], lw=1.8, ls=ls, label=label_text))
        report["methods"][method] = {
            "label": label_text, "color": colors[key], "line_style": ls,
            "n": len(rows), "safe_completions": int(counts[-1]),
            "abort_count": sum(r["outcome"] == "aborted_unrecoverable" for r in rows),
            "curve": {"time_s": grid.tolist(), "safe_completion_count": counts.tolist(),
                      "safe_completion_fraction": fraction.tolist(), "denominator": len(rows),
                      "interpolation": "right-continuous step (post)"},
            "per_seed": rows,
        }
        report["sources"].extend({k: row[k] for k in (
            "method", "suite", "seed", "result", "result_sha256", "trajectory", "trajectory_sha256")}
                                 for row in rows)
    ax.set(xlim=(0, common_horizon), ylim=(-3, 106), xticks=np.linspace(0, common_horizon, 5),
           yticks=[0, 20, 40, 60, 80, 100], xlabel="Physical execution time (s)",
           ylabel="Safe completion (%)")
    ax.grid(axis="y", color=colors["gray"], alpha=.22, lw=.5)
    fig.text(.12, .947, "Safe Completion During Unjamming", fontsize=10.5, fontweight="bold")
    fig.text(.12, .884, "Controlled MGA ablations | 10 paired seeds | empirical fractions", fontsize=8)
    legend = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.51, .112),
                        ncol=3, frameon=True, fancybox=False, edgecolor=colors["gray"],
                        facecolor="white", framealpha=1, fontsize=8, handlelength=2.3,
                        columnspacing=1.7, borderpad=.45)
    legend.get_frame().set_linewidth(.5)
    fig.text(.12, .073, "All 10 attempts remain in each denominator; aborted attempts remain failures.", fontsize=8)
    fig.text(.12, .030, "LRC = local residual correction. No post-abort execution is imputed.", fontsize=8)
    report["layout"] = {"size_inches": [5.5, 3.3], "full_horizon_s": common_horizon,
                        "legend": "bordered bottom legend", "uncertainty_bands": False}
    output.mkdir(parents=True, exist_ok=True)
    save_humanoid_appendix(fig, output, "humanoid_safe_completion", report)


def plot_humanoid_appendix(root, output, seeds):
    """Additional results from saved execution only, with task-owned trimming."""
    from genedynamics.evaluation.metrics import cvar
    output.mkdir(parents=True, exist_ok=True)
    colors, color_path = palette()
    style(colors)
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7})
    groups = {method: [load_run(root, method, "p1_force_15n", seed) for seed in seeds]
              for method, *_ in FORCE_METHODS}
    fields = ("physics_force_peak", "physics_steady_force_tracking_mae",
              "physics_force_normalized_cvar95", "physics_force_rise_time",
              "physics_force_settling_time")
    metric_sources = [
        {"path": source_path(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in (ROOT / "genedynamics/evaluation/metrics.py",
                     ROOT / "genedynamics/experiments/plugins/metrics/extractors.py")]
    force_data = {"seeds": seeds, "suite": "p1_force_15n", "methods": {}, "sources": [],
                  "palette_source": source_path(color_path),
                  "metric_sources": metric_sources,
                  "semantics": [
                      "Actual 4 ms force samples, after task-owned exclusion of unexecuted padding; no replay.",
                      "Force regulation uses force_step_success, not the fixed box's task_success field.",
                      "Steady MAE uses the frozen expected-force tracking mask, not a plot-selected window.",
                      "Small marks are individual seeds; large marks are means; identical deterministic trials overlap.",
                      "Rise/settling times are recorded metrics; retained in JSON without treating timeout values as convergence.",
                  ]}
    fig = plt.figure(figsize=(7.2, 3.35))
    ax = fig.add_axes([.105, .27, .49, .56])
    table_ax = fig.add_axes([.65, .27, .32, .56]); table_ax.axis("off")
    markers = ("o", "^", "s", "D", "P", "X")
    handles = []
    table_ax.text(0, 1, "Method", fontsize=7.5, fontweight="bold", va="top")
    table_ax.text(.70, 1, "Peak (N)\nmean $\\pm$ SD", fontsize=7.2, ha="center", va="top")
    table_ax.plot([0, 1], [.80, .80], color=colors["gray"], lw=.5)
    for index, ((method, label_text, key, _), marker) in enumerate(zip(FORCE_METHODS, markers)):
        runs = groups[method]
        summary = {field: stats([r.metrics[field] for r in runs]) for field in fields}
        summary["force_step_successes"] = sum(int(r.metrics["force_step_success"]) for r in runs)
        summary["n"] = len(runs)
        summary["per_seed"] = [{"seed": r.seed, **{field: r.metrics[field] for field in fields},
                                 "force_step_success": r.metrics["force_step_success"]} for r in runs]
        force_data["methods"][method] = summary
        force_data["sources"].extend(appendix_run_source(r) for r in runs)
        x = np.asarray(summary["physics_force_peak"]["values"])
        y = np.asarray(summary["physics_steady_force_tracking_mae"]["values"])
        ax.scatter(x, y, s=22, marker=marker, color=colors[key], alpha=.5,
                   edgecolors="white", linewidths=.3, zorder=3)
        ax.scatter(x.mean(), y.mean(), s=85, marker=marker, color=colors[key],
                   edgecolors="white", linewidths=.85, zorder=4)
        handles.append(Line2D([], [], marker=marker, color=colors[key], lw=0,
                              markersize=5, label=label_text))
        row_y = .70 - .128 * index
        table_ax.text(0, row_y, label_text, color=colors[key], fontsize=7.4, va="center")
        table_ax.text(.70, row_y, f"${x.mean():.1f} \\pm {x.std():.1f}$",
                      ha="center", fontsize=7.4, va="center")
    table_ax.set(xlim=(0, 1), ylim=(-.07, 1.05))
    ax.axvline(60, color=colors["slate"], lw=.7, ls=(0, (3, 3)))
    ax.text(60.6, .97, "60 N limit", transform=ax.get_xaxis_transform(),
            fontsize=6.6, va="top")
    ax.set(xlabel="Peak hand force (N)", ylabel="Steady force MAE (N)", xlim=(14, 73))
    ax.grid(alpha=.15, lw=.5)
    ax.margins(y=.15)
    fig.suptitle("15 N Force Regulation: Impact and Steady Tracking", x=.105, ha="left",
                 y=.98, fontsize=10, fontweight="bold")
    fig.text(.105, .9, "Each mark represents one execution. Large markers show method means.", fontsize=7.3)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.53, .042), ncol=3,
               frameon=False, columnspacing=2, fontsize=7.6)
    fig.text(.105, .026, "Lower-left is better. Deterministic ISSA/ATACOM seeds coincide.", fontsize=7)
    save_humanoid_appendix(fig, output, "humanoid_force_tradeoff", force_data)

    continuation = {"seeds": seeds, "suite": "p3_unjam", "methods": {}, "p2_nominal": {},
                    "sources": [], "palette_source": source_path(color_path),
                    "metric_sources": metric_sources,
                    "semantics": [
                        "Bars end at the last actual physical sample after task-owned success/fall trimming.",
                        "A rejection marker denotes no_revalidated_safe_candidate; the rejected action is not executed.",
                        "Certified rejections remain failures in the all-seed completion denominator.",
                        "Conditional load summaries keep completed runs and truncated rejected prefixes separate.",
                        "no_retraction is the single-switch no-LRC ablation, not physical retract removal.",
                    ]}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=.09, right=.985, top=.80, bottom=.21, wspace=.18)
    for index, (ax, (method, label_text, key)) in enumerate(zip(axes, APPENDIX_CONTINUATION_METHODS)):
        runs = [load_run(root, method, "p3_unjam", seed) for seed in seeds]
        rows = []
        for row_index, run in enumerate(runs):
            times, _ = force_trace(run)
            duration = float(times[-1])
            # Aborted canonical result files intentionally censor force metrics.
            # Their original executed physics tape remains available: evaluate
            # that prefix for a separate diagnostic, never impute a full trial.
            observed_violation = float(run.signals["safe_success_violation"])
            physical_safe = observed_violation <= 0
            complete = bool(run.metrics["task_success"])
            rejected = run.execution_status.get("reason") == "no_revalidated_safe_candidate"
            outcome = ("unsafe_execution" if not physical_safe else
                       "safe_completion" if complete else
                       "certified_rejection" if rejected else "incomplete")
            if rejected and len(run.signals["box_x"]) != run.execution_status["executed_steps"]:
                raise ValueError(f"Rejected prefix length differs from execution status: {run.trajectory_path}")
            marker = {"safe_completion": "o", "certified_rejection": "x",
                      "unsafe_execution": "^", "incomplete": "s"}[outcome]
            ax.plot([0, duration], [row_index, row_index], color=colors[key], lw=2.4, alpha=.55)
            ax.scatter(duration, row_index, marker=marker, color=colors[key], s=29,
                       linewidths=1.2, zorder=3)
            _, forces = force_trace(run)
            prefix_tail = cvar(forces / float(run.signals["f_max"]))
            if "physics_force_normalized_cvar95" in run.metrics and not np.isclose(
                    prefix_tail, run.metrics["physics_force_normalized_cvar95"]):
                raise ValueError(f"Saved tail disagrees with physical tape: {run.result_path}")
            distance, yaw = phase_trace(run)
            rows.append({"seed": run.seed, "outcome": outcome, "duration_s": duration,
                         "retained_control_steps": len(run.signals["box_x"]),
                         "execution_status": run.execution_status.get("state", "completed"),
                         "execution_reason": run.execution_status.get("reason"),
                         "physics_force_normalized_cvar95": prefix_tail,
                         "physics_force_peak": float(forces.max()),
                         "force_metric_scope": "saved_actual_execution_prefix; rejected runs are diagnostic only",
                         "goal_error_mm": float(distance[-1]),
                         "yaw_error_degrees": float(yaw[-1]),
                         "physics_max_violation": observed_violation})
        counts = {name: sum(r["outcome"] == name for r in rows) for name in
                  ("safe_completion", "certified_rejection", "unsafe_execution", "incomplete")}
        conditional = {}
        for outcome in counts:
            selected = [r for r in rows if r["outcome"] == outcome]
            if selected:
                conditional[outcome] = {field: stats([r[field] for r in selected]) for field in
                                        ("duration_s", "physics_force_normalized_cvar95", "physics_force_peak")}
        continuation["methods"][method] = {"counts": counts, "n": len(rows),
                                           "per_seed": rows, "conditional": conditional}
        continuation["sources"].extend(appendix_run_source(r) for r in runs)
        ax.set_title(f"({chr(97 + index)}) {label_text}\n{counts['safe_completion']}/{len(rows)} completed, with "
                     f"{counts['certified_rejection']} rejected.", fontsize=8.2, color=colors[key], pad=8)
        ax.set(xlim=(0, 2.05), xticks=[0, .5, 1, 1.5, 2], xlabel="Executed time (s)",
               ylim=(len(seeds) - .4, -.6))
        ax.set_yticks(range(len(seeds))); ax.set_yticklabels(seeds)
        ax.grid(axis="x", alpha=.15, lw=.5)
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
        push = [load_run(root, method, "p2_push_nominal", seed) for seed in seeds]
        continuation["p2_nominal"][method] = {
            "safe_successes": sum(int(r.metrics["safe_success"]) for r in push), "n": len(push),
            "normalized_force_tail": stats([r.metrics["physics_force_normalized_cvar95"] for r in push])}
        continuation["sources"].extend(appendix_run_source(r) for r in push)
    axes[0].set_ylabel("Seed")
    fig.suptitle("Unjamming: Completion and Safety-Triggered Stops", x=.09,
                 ha="left", y=.98, fontsize=10, fontweight="bold")
    handles = [Line2D([], [], marker=marker, color=colors["slate"], lw=0, markersize=5, label=label_text)
               for marker, label_text in (("o", "Safe completion"), ("x", "No safe candidate was found, so execution stopped."))]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.54, .035),
               ncol=2, frameon=False, fontsize=7.3)
    save_humanoid_appendix(fig, output, "humanoid_continuation", continuation)


def plot_humanoid_execution_geometry(root, output, scene_dir, seed):
    """Native context plus recorded pose/load paths; no candidate replay."""
    output.mkdir(parents=True, exist_ok=True)
    runs = [load_run(root, method, "p3_unjam", seed)
            for method, *_ in APPENDIX_CONTINUATION_METHODS]
    metadata_path = scene_dir / "metadata.json"
    native = json.loads(metadata_path.read_text())["humanoid_unjamming"]
    if native["seed"] != seed or native.get("capture_pending", False):
        raise ValueError("Native context must match the predeclared paired seed and be fully rendered")
    for key, path in (("result_sha256", runs[0].result_path),
                      ("trajectory_sha256", runs[0].trajectory_path)):
        if native[key] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("Native H1 context is stale relative to the canonical execution")
    frame = native["frames"][3]
    image_path = scene_dir / frame["image"]
    if frame["image_sha256"] != hashlib.sha256(image_path.read_bytes()).hexdigest():
        raise ValueError("Native H1 context image hash mismatch")
    colors, color_path = palette(); style(colors)
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7})
    fig = plt.figure(figsize=(7.2, 2.95))
    context = fig.add_axes([.012, .25, .285, .60])
    context.imshow(plt.imread(image_path)); context.axis("off")
    fig.text(.014, .93, "(a) Constrained Box Contact", fontsize=8.2, fontweight="bold")
    pose_ax = fig.add_axes([.395, .25, .235, .60])
    force_ax = fig.add_axes([.75, .25, .235, .60])
    fig.text(.395, .93, "(b) Executed Box Pose", fontsize=8.2, fontweight="bold")
    fig.text(.75, .93, "(c) Load and Progress", fontsize=8.2, fontweight="bold")
    origin = float(np.asarray(runs[0].signals["start_pos"]).reshape(-1)[0])
    target = float(np.asarray(runs[0].signals["target"]).reshape(-1)[0])
    target_progress = (target - origin) * 1000
    position_tolerance = float(runs[0].signals["goal_tolerance"]) * 1000
    yaw_tolerance = np.rad2deg(float(runs[0].result["config_snapshot"]["env_params"]["unjam_yaw_eps"]))
    pose_ax.add_patch(Rectangle((target_progress - position_tolerance, 0),
                               2 * position_tolerance, yaw_tolerance,
                               facecolor=colors["goal_fill"], edgecolor=colors["goal_border"],
                               lw=.6, zorder=0))
    force_ax.axvspan(target_progress - position_tolerance, target_progress + position_tolerance,
                     color=colors["goal_fill"], alpha=.45, zorder=0)
    force_ax.axhline(float(runs[0].signals["f_max"]), color=colors["slate"], ls=(0, (3, 3)), lw=.6)
    force_ax.text(.03, float(runs[0].signals["f_max"]) + 1.3, "60 N limit", fontsize=6.7,
                   transform=force_ax.get_yaxis_transform(), va="bottom")
    report = {"suite": "p3_unjam", "illustrative_seed": seed,
              "selection": "Same predeclared common seed as main figures; no method-specific selection",
              "context": {"metadata": source_path(metadata_path), "entry": "humanoid_unjamming",
                          "image": source_path(image_path), "image_sha256": frame["image_sha256"],
                          "state_index": frame["state_index"], "time_s": frame["time_seconds"],
                          "native_render_provenance": native},
              "forward_task_start_m": origin, "target_progress_mm": target_progress,
              "position_tolerance_mm": position_tolerance, "yaw_tolerance_degrees": float(yaw_tolerance),
              "methods": {}, "sources": [appendix_run_source(r) for r in runs],
              "palette_source": source_path(color_path),
              "semantics": [
                  "Native context is a verified forward-kinematic image of a saved MGA state; no simulation or controller is rerun.",
                  "Progress is actual recorded box_x minus the frozen task start_pos, not a command or planned displacement.",
                  "Yaw is the wrapped absolute error relative to the recorded target_yaw.",
                  "The pose goal rectangle is a necessary position/yaw condition, not a full task or safety certificate.",
                  "Load-progress curves pair actual control-endpoint force and actual box pose at the same 20 ms sampling; no substep pose is fabricated.",
                  "Physical safety is checked separately against the original complete 4 ms tape; the 20 ms force curve does not replace peak/safety metrics.",
                  "Task-owned evaluation excludes success padding and post-terminal suffixes. Crosses mark the last executed state before no_revalidated_safe_candidate, not the rejected action.",
                  "Lines preserve chronological order, including any progress reversal; they are executed trajectories, not fitted force-response functions.",
              ]}
    for run, (_, label_text, key), ls in zip(runs, APPENDIX_CONTINUATION_METHODS, ("-", "--", "-.")):
        signals = run.signals
        if not np.isclose(float(np.asarray(signals["start_pos"]).reshape(-1)[0]), origin):
            raise ValueError("Matched H1 examples must have the same task start")
        progress = (np.asarray(signals["box_x"], float) - origin) * 1000
        _, yaw = phase_trace(run)
        force = np.asarray(signals["force"], float)
        if progress.shape != force.shape or yaw.shape != progress.shape:
            raise ValueError("Pose and control-step force clocks are not aligned")
        physical_violation = float(signals["safe_success_violation"])
        rejected = run.execution_status.get("reason") == "no_revalidated_safe_candidate"
        completed = bool(run.metrics["task_success"])
        if rejected and len(progress) != run.execution_status["executed_steps"]:
            raise ValueError("Rejected example must retain exactly the executed prefix")
        marker = "x" if rejected else "o"
        for ax, values in ((pose_ax, yaw), (force_ax, force)):
            ax.plot(progress, values, color=colors[key], lw=1.15, ls=ls)
            ax.scatter(progress[-1], values[-1], marker=marker, color=colors[key], s=28,
                       linewidths=1.1, zorder=5)
        report["methods"][label_text] = {
            "progress_mm": progress.tolist(), "yaw_error_degrees": yaw.tolist(),
            "control_endpoint_force_N": force.tolist(),
            "time_s": ((np.arange(len(progress)) + 1) * float(signals["dt"])).tolist(),
            "actual_duration_s": len(progress) * float(signals["dt"]),
            "retained_control_steps": len(progress), "task_completed": completed,
            "physics_max_violation": physical_violation, "rejected": rejected,
            "execution_reason": run.execution_status.get("reason"),
            "final_progress_mm": float(progress[-1]), "final_goal_error_mm": float(abs(target_progress - progress[-1])),
            "final_yaw_error_degrees": float(yaw[-1]),
            "peak_physics_force_N": float(np.max(signals["physics_hand_force"])),
            "control_endpoint_force_mean_N": float(force.mean())}
    pose_ax.set(xlabel="Forward progress (mm)", ylabel="Yaw error (deg)",
                xlim=(-2, 36), ylim=(0, 5.3), xticks=[0, 15, 30], yticks=[0, 2, 4])
    force_ax.set(xlabel="Forward progress (mm)", ylabel="Control-step force (N)",
                 xlim=(-2, 36), ylim=(-3, 67), xticks=[0, 15, 30], yticks=[0, 30, 60])
    pose_ax.text(.97, .04, "pose goal", transform=pose_ax.transAxes, ha="right",
                 color=colors["goal_border"], fontsize=6.7)
    for ax in (pose_ax, force_ax):
        ax.grid(alpha=.13, lw=.5)
    handles = [Line2D([], [], color=colors[key], ls=ls, lw=1.2, label=label_text)
               for (_, label_text, key), ls in zip(APPENDIX_CONTINUATION_METHODS, ("-", "--", "-."))]
    handles.extend([Line2D([], [], color=colors["slate"], marker="o", lw=0, markersize=4.5, label="Completed"),
                    Line2D([], [], color=colors["slate"], marker="x", lw=0, markersize=5, label="Rejected")])
    legend = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.54, .035), ncol=5,
                        frameon=True, fancybox=False, facecolor="white", edgecolor=colors["gray"],
                        framealpha=1, fontsize=7.2, handlelength=1.6, columnspacing=1.1, borderpad=.4)
    legend.get_frame().set_linewidth(.5)
    save_humanoid_appendix(fig, output, "humanoid_execution_geometry", report)


def refresh_saved_scene(args):
    """Replace one embedded scene without recomputing or redrawing any curve.

    This presentation-only path deliberately needs no simulation dependencies.
    It preserves the current PDF text, vector plots and all other raster assets.
    """
    import io
    import os
    import tempfile

    import pymupdf
    from PIL import Image, ImageOps

    scene_dir = args.scene_dir or args.output_dir / "scenes"
    summary_path = args.output_dir / "humanoid_mechanism_data.json"
    summary = json.loads(summary_path.read_text())
    metadata = json.loads((scene_dir / "metadata.json").read_text())
    key = args.refresh_scene
    image_path = scene_dir / SCENES[key]
    record = metadata[image_path.stem]
    previous = summary["scene_inputs"][key]["verified_render_metadata"]
    for field in ("result", "result_sha256", "trajectory", "trajectory_sha256",
                  "seed", "state_indices", "camera", "image_size", "time_seconds"):
        if record[field] != previous[field]:
            raise ValueError(f"Appearance-only refresh changed {field}; stop: {image_path}")
    if record.get("capture_pending", True):
        raise ValueError(f"Scene capture has not completed: {image_path}")
    for field in ("result", "trajectory"):
        source = ROOT / record[field]
        if hashlib.sha256(source.read_bytes()).hexdigest() != record[field + "_sha256"]:
            raise ValueError(f"Scene {field} differs from the saved formal data: {source}")
    if hashlib.sha256(image_path.read_bytes()).hexdigest() != record["image_sha256"]:
        raise ValueError(f"Scene image hash mismatch: {image_path}")

    pdf_path = args.figure_pdf or args.output_dir / "humanoid_mechanism.pdf"
    document = pymupdf.open(pdf_path)
    if len(document) != 1:
        raise ValueError("Expected the existing one-page mechanism figure")
    page = document[0]
    layout = summary["layout"]
    bounds = (layout["force_scene_bounds_inches"] if key == "force" else
              layout["equal_panel_bounds_inches"][1 if key == "push" else 2])
    x, y, width, height = bounds
    figure_height = layout["size_inches"][1]
    expected = pymupdf.Rect(x*72, (figure_height-y-height)*72,
                           (x+width)*72, (figure_height-y)*72)
    candidates = [item for item in page.get_images(full=True)
                  if len(page.get_image_rects(item[0])) == 1
                  and max(abs(a-b) for a, b in zip(page.get_image_rects(item[0])[0], expected)) < .02]
    if len(candidates) != 1:
        raise ValueError("Current PDF layout does not uniquely match the scene; refusing to guess")
    chosen = candidates[0]
    old_text = page.get_text()
    other_images = {item[0]: hashlib.sha256(document.xref_stream(item[0])).hexdigest()
                    for item in page.get_images(full=True) if item[0] != chosen[0]}
    before = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
    # Match the existing plot's centered, aspect-preserving cover rectangle.
    with Image.open(image_path) as source:
        resized = ImageOps.fit(source.convert("RGB"), (chosen[2], chosen[3]),
                               method=Image.Resampling.LANCZOS, centering=(.5, .5))
        stream = io.BytesIO()
        resized.save(stream, format="PNG")
    page.replace_image(chosen[0], stream=stream.getvalue())
    if page.get_text() != old_text or any(
            hashlib.sha256(document.xref_stream(xref)).hexdigest() != digest
            for xref, digest in other_images.items()):
        raise ValueError("Unrelated PDF text or image changed during the scene refresh")
    after = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
    old_pixels = np.frombuffer(before.samples, dtype=np.uint8).reshape(before.height, before.width, before.n)
    new_pixels = np.frombuffer(after.samples, dtype=np.uint8).reshape(after.height, after.width, after.n)
    unchanged = np.ones(old_pixels.shape[:2], dtype=bool)
    unchanged[max(0, int(expected.y0*2)-2):int(np.ceil(expected.y1*2))+2,
              max(0, int(expected.x0*2)-2):int(np.ceil(expected.x1*2))+2] = False
    if not np.array_equal(old_pixels[unchanged], new_pixels[unchanged]):
        raise ValueError("Pixels outside the requested scene changed; refusing to save")
    with tempfile.NamedTemporaryFile(dir=pdf_path.parent, suffix=".pdf", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    document.save(temporary_path, garbage=3, deflate=True)
    if args.figure_pdf is None:
        page.get_pixmap(matrix=pymupdf.Matrix(350/72, 350/72), alpha=False).save(pdf_path.with_suffix(".png"))
    document.close()
    os.replace(temporary_path, pdf_path)
    summary["scene_inputs"][key]["verified_render_metadata"] = record
    summary["scene_inputs"][key]["path"] = source_path(image_path)
    summary["scene_refresh"] = {
        "panel": key, "source_scene_sha256": record["image_sha256"],
        "preserved": "Current PDF text, vector curves, numerical evidence, other images, and all pixels outside the scene rectangle",
        "data_recomputed": False, "outside_scene_pixel_equality_verified_at_144_dpi": True,
    }
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Refreshed only {key}: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "results/humanoid/push_to_line")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/mga/paper_figures")
    parser.add_argument("--scene-dir", type=Path)
    parser.add_argument("--refresh-scene", choices=tuple(SCENES),
                        help="Replace only this scene in the existing PDF, preserving current plots and text")
    parser.add_argument("--figure-pdf", type=Path,
                        help="With --refresh-scene, update this existing PDF instead of the report copy")
    parser.add_argument("--seed", type=int, default=0, help="Common illustrative seed; never best per method")
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("0-9"))
    parser.add_argument("--require-scenes", action="store_true", help="Fail rather than emit scene placeholders")
    parser.add_argument("--appendix", action="store_true", help="Write additional-result figures only; preserve the main figure")
    parser.add_argument("--appendix-geometry-only", action="store_true", help="With --appendix, render only simulation-linked execution geometry")
    parser.add_argument("--completion-curve", action="store_true", help="Write only the paired H1 unjamming safe-completion preview; requires seeds 0-9")
    args = parser.parse_args()
    if args.refresh_scene:
        if args.appendix or args.appendix_geometry_only or args.completion_curve:
            parser.error("--refresh-scene is a separate presentation-only operation")
        refresh_saved_scene(args)
        return
    if args.figure_pdf:
        parser.error("--figure-pdf requires --refresh-scene")
    if args.completion_curve:
        if args.appendix or args.appendix_geometry_only:
            parser.error("--completion-curve is separate from --appendix")
        if args.seeds != list(range(10)):
            parser.error("--completion-curve requires all paired seeds 0-9")
        plot_humanoid_safe_completion(args.data_root, args.output_dir, args.seeds)
        return
    if args.appendix_geometry_only and not args.appendix:
        parser.error("--appendix-geometry-only requires --appendix")
    if args.seed not in args.seeds:
        parser.error("--seed must be included in --seeds")
    if args.appendix:
        appendix_output = args.output_dir / "appendix"
        if not args.appendix_geometry_only:
            plot_humanoid_appendix(args.data_root, appendix_output, args.seeds)
        plot_humanoid_execution_geometry(args.data_root, appendix_output,
                                         args.scene_dir or appendix_output / "scenes", args.seed)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene_dir = args.scene_dir or args.output_dir / "scenes"
    colors, colors_path = palette()
    requests = {(method, "p1_force_15n") for method, *_ in FORCE_METHODS}
    requests.update((method, "p3_unjam") for method, *_ in PHASE_METHODS)
    requests.add(("main/mga", "p2_push_nominal"))
    groups = {(method, suite): [load_run(args.data_root, method, suite, seed)
                                for seed in args.seeds]
              for method, suite in sorted(requests)}
    scene_metadata = validate_scenes(groups, scene_dir, args.seed, args.require_scenes)
    style(colors)
    fig = create_figure(groups, scene_dir, args.seed, colors, args.require_scenes, scene_metadata)
    panel_bounds = fig._mga_panel_bounds_inches
    force_plot_bounds = fig._mga_force_plot_bounds_inches
    force_scene_bounds = fig._mga_force_scene_bounds_inches
    arrow_evidence = fig._mga_arrow_evidence
    force_highlight = fig._mga_force_highlight
    stem = args.output_dir / "humanoid_mechanism"
    fig.savefig(stem.with_suffix(".pdf"), dpi=400, facecolor="white", pad_inches=0.025)
    fig.savefig(stem.with_suffix(".png"), dpi=350, facecolor="white", pad_inches=0.025)
    plt.close(fig)

    summary = {
        "figure": "Humanoid force regulation, pushing, and safe continuation",
        "illustrative_seed": args.seed, "aggregate_seeds": args.seeds,
        "layout": {"size_inches": [4.00, 3.45],
                   "equal_panel_bounds_inches": panel_bounds,
                   "force_plot_bounds_inches": force_plot_bounds,
                   "force_scene_bounds_inches": force_scene_bounds,
                   "panel_size_inches": [1.77, 1.31],
                   "font_family": "DejaVu Sans",
                   "views": ["side/oblique over force plot", "side/oblique", "top", "phase portrait"]},
        "palette_source": source_path(colors_path), "palette": colors,
        "scene_direction_arrows": arrow_evidence,
        "force_highlight": force_highlight,
        "semantics": {
            "force": "Actual post-substep hand force; no smoothing or padded suffix",
            "force_legend": "Full method names in the bottom shared legend; force plot is below the scene, not an inset",
            "force_reference": "Recorded task_force_reference repeated across actual physics substeps; identical clocks/reference profiles verified across plotted methods",
            "phase": "Executed box states; bold common illustrative seed, faint remaining seeds",
            "phase_x": "Absolute box position error in the forward axis, abs(target_x - box_x), in millimeters",
            "phase_target": "Task position/yaw tolerances only, not a complete safety certificate",
            "stop_marker": "No revalidated safe candidate; rejected action was not executed",
            "no_lrc_source": "ablation/no_retraction is the recorded no-LRC ablation",
            "simulation": "Only recorded first and final unpadded states; no dynamic replay or altered rollout",
            "scene_stage_legends": "B/C keys read the actual native box-outline colors from scene metadata; the same colors identify initial/final robot silhouettes",
            "scene_time_labels": "Min/max of each verified scene time_seconds list; these are ghost-frame times (C outlines also include state 0)",
            "peak_std": "Population standard deviation of recorded per-run physics_force_peak",
        },
        "force_methods": [{"method": method, "label": label, "color": colors[color],
                           "line_style": line_style}
                          for method, label, color, line_style in FORCE_METHODS],
        "scene_inputs": {key: {"path": source_path(scene_dir / name),
                              "available": (scene_dir / name).exists(),
                              "verified_render_metadata": scene_metadata.get(key)}
                         for key, name in SCENES.items()},
        "force_peak_N": {}, "fixed_stance": {}, "unjamming": {}, "sources": [],
    }
    for method, *_ in FORCE_METHODS:
        runs = groups[(method, "p1_force_15n")]
        summary["force_peak_N"][method] = stats([r.metrics["physics_force_peak"] for r in runs])
    push_runs = groups[("main/mga", "p2_push_nominal")]
    summary["fixed_stance"] = {
        "successes": sum(float(r.metrics["task_success"]) for r in push_runs),
        "n": len(push_runs),
        "normalized_force_tail": stats([r.metrics["physics_force_normalized_cvar95"] for r in push_runs]),
    }
    for method, *_ in PHASE_METHODS:
        runs = groups[(method, "p3_unjam")]
        summary["unjamming"][method] = {
            "task_successes": sum(float(r.metrics["task_success"]) for r in runs),
            "n": len(runs), "endpoints": [
                {"seed": r.seed, "distance_mm": float(phase_trace(r)[0][-1]),
                 "yaw_deg": float(phase_trace(r)[1][-1]),
                 "execution_reason": r.execution_status.get("reason"),
                 "retained_steps": len(r.signals["box_x"])} for r in runs],
        }
    for runs in groups.values():
        summary["sources"].extend({
            "method": r.method, "suite": r.suite, "seed": r.seed,
            "result": source_path(r.result_path), "trajectory": source_path(r.trajectory_path),
        } for r in runs)
    (args.output_dir / "humanoid_mechanism_data.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(stem.with_suffix(".pdf"))
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
