"""Revise the published Figure 3 on a copy of the paper page.

Scene renders are not in this checkout, so the manuscript page is edited in
place on a copy: title case, one framed key under (b), Raw peak on (c),
the Figure 2 socket color on (e), and a frame under (f).
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "output"
INK = (0x3A / 255, 0x44 / 255, 0x55 / 255)
RULE = (0.604, 0.604, 0.604)
LABEL = 5.0
TITLE = 5.7
# Panel (e) sits a little lower. (f) and (g) follow so the two gaps stay equal.
E_DROP = 5.0
F_SHIFT = 0.6 - E_DROP
G_DROP = 8.0 + E_DROP
TEAL = np.array([176, 214, 210], np.float32)

TITLES = {
    "(a) Bumpy-surface scanning: shape and contact":
        "(a) Bumpy Rigid Surface Replay",
    "(b) Hybrid map and paths (3-D contact coverage)":
        "(b) Hybrid Stripe Rigid-Soft Pattern Execution Coverage (Shown On 3D Plane)",
    "(c) Hybrid response (3-pt mean)":
        "(c) Force\u2013Stiffness Response Result Of (b)",
    "(d) LRC on a curved surface":
        "(d) LRC On A Soft Curved Surface",
    "(e) Recorded contact progression":
        "(e) Pose-OOD Contact Progression Replay",
    "(f) Executed wrench response":
        "(f) Unfiltered Wrench Utilization\nResponse Result Of (e)",
    "(g) Logged plan selection":
        "(g) Logged Plan Selection For RL Prior Of (e)",
}
CALLOUTS = {
    "limit": "Limit",
    "target": "Target",
}


def font_paths():
    import matplotlib
    base = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    return base / "DejaVuSans.ttf", base / "DejaVuSans-Bold.ttf"


def spans(page):
    found = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                found.append(span)
    return found


def redact(page, rect):
    page.add_redact_annot(rect + (-0.3, -0.3, 0.3, 0.3), fill=(1, 1, 1))


def _largest_component(seed):
    height, width = seed.shape
    seen = np.zeros((height, width), dtype=bool)
    best = []
    ys, xs = np.where(seed)
    for y, x in zip(ys, xs):
        if seen[y, x]:
            continue
        stack = [(int(y), int(x))]
        seen[y, x] = True
        component = []
        while stack:
            cy, cx = stack.pop()
            component.append((cy, cx))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < height and 0 <= nx < width and seed[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if len(component) > len(best):
            best = component
    return best


def paint_socket(image):
    """Paint the whole socket prism in Figure 2's mint, keeping the mesh edges."""
    array = np.asarray(image.convert("RGB")).astype(np.float32)
    red, green, blue = array[:, :, 0], array[:, :, 1], array[:, :, 2]
    height, width = blue.shape
    seed = (blue > 108) & ((blue - green) > 30) & ((blue - red) > 42) & (red < 90)
    seed[:, :115] = False
    seed[:, 175:] = False
    component = _largest_component(seed)
    if not component:
        return image
    ys = [point[0] for point in component]
    xs = [point[1] for point in component]
    y0, y1, x0, x1 = min(ys), max(ys), min(xs), max(xs)
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    # Include the top face, which is darker than the sides, but stop at the base.
    window = (rows >= y0 - 16) & (rows <= y1 + 1) & (cols >= x0 - 3) & (cols <= x1 + 3)
    face = (blue > 64) & ((blue - red) > 18) & ((blue - green) > 12) & (red < 115) & window
    seen = np.zeros((height, width), dtype=bool)
    stack = [(int(y), int(x)) for y, x in component]
    for y, x in stack:
        seen[y, x] = True
    index = 0
    while index < len(stack):
        cy, cx = stack[index]
        index += 1
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < height and 0 <= nx < width and face[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                stack.append((ny, nx))
    mask = seen
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    # Keep a little face shading, but do not let the top face fall back to blue.
    factor = np.clip(luminance / float(luminance[mask].mean()), 0.9, 1.06)
    array[mask] = np.clip(TEAL * factor[mask][:, None], 0, 255)
    return Image.fromarray(array.astype(np.uint8))


def replace_sockets(doc, page):
    for xref, _smask, width, height, *_rest in page.get_images():
        if (width, height) != (290, 193):
            continue
        info = doc.extract_image(xref)
        image = Image.open(io.BytesIO(info["image"]))
        painted = paint_socket(image)
        buffer = io.BytesIO()
        painted.save(buffer, format="PNG")
        page.replace_image(xref, stream=buffer.getvalue())


def legend_spans(page, frame):
    x0, y0, x1, y1 = frame
    chosen = []
    for span in spans(page):
        box = pymupdf.Rect(span["bbox"])
        if box.x0 >= x0 - 1 and box.x1 <= x1 + 1 and box.y0 >= y0 - 1 and box.y1 <= y1 + 1:
            chosen.append(span)
    return chosen


def put(page, point, text, size, fontname, color=INK, rotate=0):
    page.insert_text(point, text, fontname=fontname, fontsize=size, color=color, rotate=rotate)


def put_rotated_axis(page, span):
    """Keep the published upward direction, and match the horizontal axis size."""
    box = pymupdf.Rect(span["bbox"])
    scale = LABEL / 5.45
    size = span["size"] * scale
    center = (box.y0 + box.y1) / 2
    origin = (span["origin"][0], center + box.height * scale / 2)
    put(page, origin, span["text"], size, "body", color_of(span), rotate=90)


def color_of(span):
    value = int(span["color"])
    return ((value >> 16) & 255) / 255, ((value >> 8) & 255) / 255, (value & 255) / 255


def colorbar_stream(name):
    # Sampled from the published stiffness (purple to teal) and force (sand to brown) bars.
    if name == "stiff":
        stops = [(227, 221, 239), (176, 214, 210)]
    else:
        stops = [(255, 236, 214), (176, 96, 40)]
    width, height = 120, 8
    row = np.zeros((height, width, 3), np.uint8)
    for index in range(width):
        blend = index / (width - 1)
        color = [
            int(stops[0][channel] + (stops[1][channel] - stops[0][channel]) * blend)
            for channel in range(3)
        ]
        row[:, index] = color
    buffer = io.BytesIO()
    Image.fromarray(row).save(buffer, format="PNG")
    return buffer.getvalue()


def latest_wrench_runs():
    """Seed-0 Sensing-OOD executions from the 2026-09-26 archive."""
    root = ROOT / "peg_insert_completed_20260926/results/arm/peg_insert"
    specs = (
        ("main/mga", "MGA", "#1F7A78", "-", 0.72),
        ("ablation/no_rl_prior", "w/o prior", "#4B2E83", "--", 0.62),
        ("baseline/dial", "DIAL", "#D9660E", ":", 0.62),
        ("baseline/issa", "ISSA", "#3A4655", (0, (2.5, 1)), 0.62),
    )
    runs = []
    for method, label, color, style, width in specs:
        path = root / method / "level_ood_sensing/seed_0/trajectory/trajectory.json"
        signals = json.loads(path.read_text())["task_signals"]
        rho = np.asarray(signals["rho"], float)
        dt = float(signals["dt"])
        time = (np.arange(len(rho)) + 1) * dt
        hit = np.flatnonzero(np.asarray(signals["success"], float) > 0.5)
        runs.append({
            "label": label, "color": color, "style": style, "width": width,
            "time": time, "rho": rho, "peak": float(rho.max()),
            "done": int(hit[0]) if hit.size else None,
        })
    return runs


def surface_raw_peaks():
    """Raw normal-force maxima for every algorithm on the panel (c) example."""
    root = ROOT / "results/arm/surface_scan"
    methods = (
        ("main/mga", "MGA", (0.122, 0.478, 0.471), None),
        ("baseline/mppi", "MPPI", (0.361, 0.561, 0.180), "[2.2 1.0] 0"),
        ("baseline/dial", "DIAL", (0.851, 0.400, 0.055), "[0.7 1.1] 0"),
        ("baseline/pegasusflow", "PegasusFlow", (0.322, 0.384, 0.478), "[3.2 0.8 0.6 0.8] 0"),
        ("baseline/atacom", "ATACOM", (0.294, 0.180, 0.514), "[2.4 0.9 0.6 0.9] 0"),
        ("baseline/issa", "ISSA", (0.227, 0.275, 0.333), "[1.6 0.7] 0"),
    )
    rows = []
    for method, label, color, dashes in methods:
        path = root / method / "level_hybrid_stripes/seed_0/trajectory/trajectory.json"
        force = np.asarray(json.loads(path.read_text())["task_signals"]["force"], float)
        rows.append({"label": label, "peak": f"{float(force.max()):.1f}",
                     "color": color, "dashes": dashes})
    return rows


def render_wrench(runs, height=46.0):
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans", "font.size": 5,
        "axes.labelsize": 5, "xtick.labelsize": 5, "ytick.labelsize": 5,
        "axes.linewidth": 0.45, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.size": 1.6, "ytick.major.size": 1.6,
        "text.color": "#3A4655", "axes.labelcolor": "#3A4655",
        "axes.edgecolor": "#3A4655", "xtick.color": "#3A4655", "ytick.color": "#3A4655",
    })
    # Vector PDF, same 5 pt type as panels (c) and (d). A short bottom margin
    # puts the tick labels against the lower edge of the panel.
    fig, ax = plt.subplots(figsize=(141 / 72, height / 72), dpi=200)
    top_pad = 6.4
    fig.subplots_adjust(left=0.16, right=0.985, bottom=12.4 / height, top=1 - top_pad / height)
    for run in runs:
        line, = ax.plot(run["time"], run["rho"], color=run["color"], ls=run["style"],
                        lw=run["width"], alpha=1, zorder=5 if run["label"] == "MGA" else 3)
        if run["label"] == "MGA":
            line.set_path_effects([
                patheffects.Stroke(linewidth=1.0, foreground="white"), patheffects.Normal()])
        if run["done"] is not None:
            ax.plot(run["time"][run["done"]], run["rho"][run["done"]], "o", ms=1.6,
                    color=run["color"], markerfacecolor="white", markeredgewidth=0.45, zorder=6)
    ax.axhline(1, color="#D98A4A", ls=":", lw=0.55)
    ax.text(1.27, 1.008, "Limit", ha="right", va="bottom", fontsize=5, color="#3A4655", clip_on=False)
    ax.set(xlim=(0, 1.28), ylim=(0, 1.12), xticks=[0, 0.4, 0.8, 1.2], yticks=[0, 0.5, 1],
           ylabel=r"True $\rho(t)$")
    ax.set_xlabel("Time (s)", labelpad=1.2)
    ax.grid(axis="y", color="#EEEEEE", lw=0.35)
    ax.tick_params(pad=0.6)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="pdf")
    plt.close(fig)
    return buffer.getvalue()


def legend_swatch(page, x0, y, color, dashes=None, length=7.8):
    """Short line sample, matching the stroke used in panel (d)'s legend."""
    kwargs = {"color": color, "width": 0.84}
    if dashes:
        kwargs["dashes"] = dashes
    page.draw_line((x0, y), (x0 + length, y), **kwargs)


def draw_wrench(page, measure):
    """Replace panel (f) with the latest seed-0 curves and a strip below the ticks."""
    runs = latest_wrench_runs()
    # Cover the old curves and the old peak numbers, and stop short of title (g).
    # The second title line occupies the old top of this panel, so the curves start lower.
    plot_top = 144.0 - F_SHIFT + 7.6
    cover_top = min(136.0, 144.0 - F_SHIFT)
    page.draw_rect(pymupdf.Rect(363.5, cover_top, 504.1, 213.5), color=(1, 1, 1), fill=(1, 1, 1), width=0)
    plot = pymupdf.Rect(363.5, plot_top, 504.1, 190.0 - F_SHIFT)
    source = pymupdf.open(stream=render_wrench(runs, plot.height), filetype="pdf")
    page.show_pdf_page(plot, source, 0)
    # Sit the legend on the bottom of the panel. Its lower padding matches panel (c).
    legend_bottom = (204.6 - F_SHIFT) + 2.8
    box = pymupdf.Rect(364.5, 191.6 - F_SHIFT, 502.5, legend_bottom)
    page.draw_rect(box, color=RULE, fill=(1, 1, 1), width=0.46)
    put(page, (367.4, 197.4 - F_SHIFT), "Raw", LABEL, "body")
    put(page, (367.4, 204.6 - F_SHIFT), "Peak", LABEL, "body")
    pairs = ((runs[0], runs[2]), (runs[1], runs[3]))
    for row, (left, right) in enumerate(pairs):
        baseline = (197.4 if row == 0 else 204.6) - F_SHIFT
        for run, x in ((left, 404), (right, 470)):
            color = tuple(int(run["color"][i:i + 2], 16) / 255 for i in (1, 3, 5))
            value = f"{run['peak']:.3f}"
            style = run["style"]
            if style == "--":
                dashes = "[2.4 1.1] 0"
            elif style == ":":
                dashes = "[0.5 0.9] 0"
            elif isinstance(style, tuple):
                dashes = "[2.5 1.0] 0"
            else:
                dashes = None
            legend_swatch(page, x - 9.4, baseline - 1.55, color, dashes)
            put(page, (x, baseline), run["label"], LABEL, "body", color)
            name_width = measure.text_length(run["label"], fontsize=LABEL)
            put(page, (x + name_width + 2.2, baseline), value, LABEL, "body", color)
    return runs


def draw_response_legend(page, measure):
    """Two rows of three algorithms, each name followed by its peak, as in panel (f)."""
    rows = surface_raw_peaks()
    # Clear the old two-row legend without reaching panel (d).
    # The old legend frame sticks out above this box, as short rules beside "Time (s)".
    page.draw_rect(pymupdf.Rect(128.0, 263.5, 248.0, 281.2), color=(1, 1, 1), fill=(1, 1, 1), width=0)
    swatch = 7.8
    swatch_gap = 1.6
    name_gap = 2.2
    col_gap = 4.6
    bands = (rows[:3], rows[3:])

    def entry_width(item):
        return (swatch + swatch_gap
                + measure.text_length(item["label"], fontsize=LABEL) + name_gap
                + measure.text_length(item["peak"], fontsize=LABEL))

    columns = [max(entry_width(bands[0][index]), entry_width(bands[1][index])) for index in range(3)]
    label_width = max(measure.text_length(word, fontsize=LABEL) for word in ("Raw", "Peak")) + 4.0
    x0, name_y, value_y = 112.0, 270.6, 277.8
    right = x0 + label_width + sum(columns) + col_gap * (len(columns) - 1) + 3.0
    page.draw_rect(pymupdf.Rect(x0, 264.8, right, 280.6), color=RULE, fill=(1, 1, 1), width=0.46)
    put(page, (x0 + 2.0, name_y), "Raw", LABEL, "body")
    put(page, (x0 + 2.0, value_y), "Peak", LABEL, "body")
    for baseline, items in ((name_y, bands[0]), (value_y, bands[1])):
        cursor = x0 + label_width
        for item, width in zip(items, columns):
            legend_swatch(page, cursor, baseline - 1.55, item["color"], item["dashes"], swatch)
            name_x = cursor + swatch + swatch_gap
            put(page, (name_x, baseline), item["label"], LABEL, "body", item["color"])
            name_width = measure.text_length(item["label"], fontsize=LABEL)
            put(page, (name_x + name_width + name_gap, baseline), item["peak"], LABEL, "body", item["color"])
            cursor += width + col_gap
    return right


def _trend(values):
    values = np.asarray(values, float)
    trend = values.copy()
    trend[1:-1] = (values[:-2] + values[1:-1] + values[2:]) / 3
    return trend


def _series_pdf(series, xlim, ylim, width, height, extras):
    """Curves only, mapped 1:1 onto a page rectangle of the given size in points."""
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    fig, ax = plt.subplots(figsize=(width / 72, height / 72))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_axis_off()
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    for item in series:
        if item.get("raw") is not None:
            ax.plot(item["time"], item["raw"], color=item["color"], lw=0.28, alpha=0.45,
                    zorder=2 if item["label"] == "MGA" else 1)
        line, = ax.plot(item["time"], item["trend"], color=item["color"], ls=item["ls"],
                        lw=0.78 if item["label"] == "MGA" else 0.68,
                        zorder=5 if item["label"] == "MGA" else 3)
        if item["label"] == "MGA":
            line.set_path_effects([
                patheffects.Stroke(linewidth=1.0, foreground="white"), patheffects.Normal()])
        if item.get("marker") is not None:
            ax.plot(item["marker"][0], item["marker"][1], marker="^", ls="none", ms=1.6,
                    color=item["color"], markeredgecolor="white", markeredgewidth=0.3, zorder=6)
    for extra in extras:
        ax.plot(extra["x"], extra["y"], color=extra["color"], ls=extra["ls"], lw=extra["lw"], zorder=2)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="pdf", transparent=True)
    plt.close(fig)
    return buffer.getvalue()


def place_d_legend(page, spans, c_right):
    """Raise panel (d)'s legend so its top edge lines up with panel (c)."""
    lift = 2.0
    shift = max(0.0, c_right + 4.0 - 279.69)
    page.draw_rect(pymupdf.Rect(max(278.5, c_right + 1.0), 264.4, 370.0, 290.0),
                   color=(1, 1, 1), fill=(1, 1, 1), width=0)
    page.draw_rect(pymupdf.Rect(279.69 + shift, 266.8 - lift, 339.28 + shift, 275.3 - lift),
                   color=RULE, fill=(1, 1, 1), width=0.46)
    y = 270.62 - lift
    page.draw_line((281.51 + shift, y), (289.30 + shift, y), color=(0.122, 0.478, 0.471), width=0.84)
    page.draw_line((306.92 + shift, y), (314.72 + shift, y), color=(0.851, 0.400, 0.055), width=0.84,
                   dashes="[3.1 1.34] 0")
    for span in spans:
        origin = span["origin"]
        put(page, (origin[0] + shift, origin[1] - lift), span["text"], LABEL, "body", color_of(span))


def draw_clear_response(page, callouts):
    """Redraw panel (c) so every method stays readable and MGA stays in front."""
    root = ROOT / "results/arm/surface_scan"
    specs = (
        ("main/mga", "MGA", (0.122, 0.478, 0.471), "-"),
        ("baseline/mppi", "MPPI", (0.361, 0.561, 0.180), (0, (2.2, 1.0))),
        ("baseline/dial", "DIAL", (0.851, 0.400, 0.055), (0, (0.7, 1.1))),
        ("baseline/pegasusflow", "PegasusFlow", (0.322, 0.384, 0.478), (0, (3.2, 0.8, 0.6, 0.8))),
        ("baseline/atacom", "ATACOM", (0.294, 0.180, 0.514), (0, (2.4, 0.9, 0.6, 0.9))),
        ("baseline/issa", "ISSA", (0.227, 0.275, 0.333), (0, (1.6, 0.7))),
    )
    force_rows, stiff_rows = [], []
    target = limit = None
    for method, label, color, style in specs:
        folder = root / method / "level_hybrid_stripes/seed_0"
        signals = json.loads((folder / "trajectory/trajectory.json").read_text())["task_signals"]
        result = json.loads((folder / "results.json").read_text())
        dt = float(result["config_snapshot"]["env_params"]["dt"])
        force = np.asarray(signals["force"], float)
        stiff = np.asarray(signals["normal_stiffness"], float) / 1000
        time = (np.arange(len(force)) + 1) * dt
        peak = int(force.argmax())
        force_rows.append({"label": label, "color": color, "ls": style, "time": time,
                           "raw": force, "trend": _trend(force), "marker": (time[peak], force[peak])})
        stiff_rows.append({"label": label, "color": color, "ls": style, "time": time,
                           "raw": stiff, "trend": _trend(stiff)})
        if label == "MGA":
            target = (time, np.asarray(signals["force_des"], float))
            limit = float(signals["f_max"])
    # Data rectangles of the published axes. y grows downward.
    force_box = pymupdf.Rect(135.9, 207.17, 227.1, 225.8)
    stiff_box = pymupdf.Rect(135.9, 230.69, 227.1, 249.3)
    for box in (force_box, stiff_box):
        page.draw_rect(box + (0.4, 0.35, -0.3, -0.3), color=(1, 1, 1), fill=(1, 1, 1), width=0)
    others = [row for row in force_rows if row["label"] != "MGA"] + [force_rows[0]]
    force_pdf = _series_pdf(
        others, (0, force_rows[0]["time"][-1]), (0, 65), force_box.width, force_box.height,
        [{"x": target[0], "y": target[1], "color": "#9AA3AD", "ls": "--", "lw": 0.45},
         {"x": [0, force_rows[0]["time"][-1]], "y": [limit, limit], "color": "#3A4655", "ls": (0, (2, 2)), "lw": 0.45}],
    )
    stiff_others = [row for row in stiff_rows if row["label"] != "MGA"] + [stiff_rows[0]]
    stiff_pdf = _series_pdf(
        stiff_others, (0, stiff_rows[0]["time"][-1]), (0.35, 1.35), stiff_box.width, stiff_box.height, [],
    )
    page.show_pdf_page(force_box, pymupdf.open(stream=force_pdf, filetype="pdf"), 0)
    page.show_pdf_page(stiff_box, pymupdf.open(stream=stiff_pdf, filetype="pdf"), 0)
    for span in callouts:
        if pymupdf.Rect(span["bbox"]).x0 > 360:
            continue
        put(page, span["origin"], CALLOUTS[span["text"]], LABEL, "body", color_of(span))


def place_event_frames(page, doc):
    """Widen the three closeups so their right edge matches panel (f)."""
    frames = []
    for info in page.get_image_info(xrefs=True):
        box = pymupdf.Rect(info["bbox"])
        if box.x0 > 360 and 90 < box.y0 < 110 and box.width < 55:
            frames.append((box.y0, box.x0, info["xref"], box))
    frames.sort()
    # The source page can report more than one copy. Keep the published row only.
    published_top = frames[0][0]
    frames = [item for item in frames if abs(item[0] - published_top) < 1.0]
    frames.sort(key=lambda item: item[1])
    eyelevel = [OUT / "e_eyelevel" / name for name in ("initial.png", "contact.png", "complete.png")]
    if all(path.exists() for path in eyelevel):
        streams = [path.read_bytes() for path in eyelevel]
    else:
        streams = [doc.extract_image(xref)["image"] for _, _, xref, _box in frames]
    # Stay inside the figure's right edge instead of sitting flush against it.
    right_edge = 501.0
    left_edge = frames[0][3].x0
    gap = 1.5
    width = (right_edge - left_edge - gap * (len(streams) - 1)) / len(streams)
    sample = Image.open(io.BytesIO(streams[0]))
    height = width * sample.height / sample.width
    origin_top = frames[0][3].y0
    top = origin_top + E_DROP
    page.draw_rect(pymupdf.Rect(left_edge - 0.4, origin_top - 0.4, right_edge + 0.4, top + height + 0.6),
                   color=(1, 1, 1), fill=(1, 1, 1), width=0)
    placed = []
    for index, stream in enumerate(streams):
        rect = pymupdf.Rect(left_edge + index * (width + gap), top,
                            left_edge + index * (width + gap) + width, top + height)
        page.insert_image(rect, stream=stream)
        placed.append(rect)
    return placed


def paint_event_captions(page, measure, captions):
    """Match panel (a): 4.27 pt white type on an 80% dark chip at the lower left."""
    size = 4.27
    badge = (0.1372549, 0.2470588, 0.3333333)
    for rect, text in captions:
        width = measure.text_length(text, fontsize=size)
        # The chip's left edge sits on the image edge. Text keeps the same inset as panel (a).
        text_x = rect.x0 + 0.94
        chip = pymupdf.Rect(rect.x0, rect.y1 - 5.94, text_x + width + 1.12, rect.y1)
        chip.x1 = min(chip.x1, rect.x1)
        page.draw_rect(chip, color=None, fill=badge, fill_opacity=0.8, width=0)
        put(page, (text_x, rect.y1 - 1.82), text, size, "body", (1, 1, 1))


def enlarge_plan_legend(page):
    """Keep Complete inside the panel (g) frame after the labels grow to 5 pt."""
    right = 485.6
    # Remove only the published right stroke. The new "Complete" ends near x=482.
    page.draw_rect(pymupdf.Rect(480.55, 259.2, 481.4, 274.6), color=(1, 1, 1), fill=(1, 1, 1), width=0)
    page.draw_line((480.5, 259.5), (right, 259.5), color=RULE, width=0.42)
    page.draw_line((right, 259.5), (right, 274.32), color=RULE, width=0.42)
    page.draw_line((480.5, 274.32), (right, 274.32), color=RULE, width=0.42)


def extend_complete_legend(page, measure):
    """Redraw Complete inside a wider frame. The 5 pt label no longer meets the stroke."""
    text = "Complete"
    text_width = measure.text_length(text, fontsize=LABEL)
    # Original label origin sits just left of the old right stroke.
    origin = (454.0, 265.0)
    page.draw_rect(pymupdf.Rect(452.0, 260.2, 481.6, 267.4), color=(1, 1, 1), fill=(1, 1, 1), width=0)
    put(page, origin, text, LABEL, "body")
    right = origin[0] + text_width + 4
    page.draw_rect(pymupdf.Rect(480.4, 260.0, 482.2, 273.8), color=(1, 1, 1), fill=(1, 1, 1), width=0)
    page.draw_line((481.0, 259.5), (right, 259.5), color=RULE, width=0.46)
    page.draw_line((right, 259.5), (right, 274.3), color=RULE, width=0.46)
    page.draw_line((481.0, 274.3), (right, 274.3), color=RULE, width=0.46)


def draw_key_fixed(page, measure):
    size = LABEL

    def width(text, used=size):
        return measure.text_length(text, fontsize=used)

    bar_w = 26
    small_gap = 3.0
    # Two group gaps are the slack. The frame must end at the trajectory tiles.
    group_gap = 4.0
    items = []
    items += [("text", "Nominal k (kN/m)"), ("gap", small_gap), ("bar", "stiff", bar_w),
              ("gap", small_gap), ("text", "["), ("gap", 1.0), ("text", "2 (Soft)"),
              ("gap", 3.6), ("text", "8 (Hard)"), ("gap", 1.0), ("text", "]"),
              ("gap", group_gap), ("fn",), ("gap", small_gap), ("bar", "force", bar_w),
              ("gap", small_gap), ("text", "["), ("gap", 1.0), ("text", "0.0"), ("gap", 3.2),
              ("text", "0.5"), ("gap", 3.2), ("text", "1.0"), ("gap", 1.0), ("text", "]"),
              ("gap", group_gap), ("dash", 8),
              ("gap", 1.6), ("text", "Ref.")]

    def item_width(item):
        kind = item[0]
        if kind == "text":
            return width(item[1])
        if kind == "gap":
            return item[1]
        if kind == "bar":
            return item[2]
        if kind == "dash":
            return item[1]
        return (
            width("F") + width("n", size * 0.72) + width("/")
            + width("F") + width("max", size * 0.72)
        )

    total = sum(item_width(item) for item in items)
    x0, baseline = 111.6, 194.95
    # Right edge of the four trajectory tiles. The key must stay inside that width.
    plot_right = 357.0
    right = min(x0 + total + 7.2, plot_right)
    page.draw_rect(
        pymupdf.Rect(x0, 190.2, right, 196.85),
        color=RULE, fill=(1, 1, 1), width=0.46,
    )
    cursor = x0 + 3.6
    for item in items:
        kind = item[0]
        if kind == "gap":
            cursor += item[1]
        elif kind == "text":
            put(page, (cursor, baseline), item[1], size, "body")
            cursor += width(item[1])
        elif kind == "bar":
            rect = pymupdf.Rect(cursor, baseline - 2.55, cursor + item[2], baseline - 0.85)
            page.insert_image(rect, stream=colorbar_stream(item[1]))
            cursor += item[2]
        elif kind == "dash":
            page.draw_line(
                (cursor, baseline - 1.5), (cursor + item[1], baseline - 1.5),
                color=INK, width=0.55, dashes="[1.6 0.8] 0",
            )
            cursor += item[1]
        else:
            put(page, (cursor, baseline), "F", size, "body")
            cursor += width("F")
            put(page, (cursor, baseline + 0.8), "n", size * 0.72, "body")
            cursor += width("n", size * 0.72)
            put(page, (cursor, baseline), "/", size, "body")
            cursor += width("/")
            put(page, (cursor, baseline), "F", size, "body")
            cursor += width("F")
            put(page, (cursor, baseline + 0.8), "max", size * 0.72, "body")
            cursor += width("max", size * 0.72)


def shift_plan_chart(page, drop):
    """Move the panel (g) bars down and drop the legend that used to sit under them."""
    source = pymupdf.open(next(ROOT.glob("_ICLR_2027__Manifold*.pdf")))
    clip = pymupdf.Rect(361.0, 218.2, 505.0, 257.6)
    # Stay below panel (f)'s legend after that panel moves down with (e).
    page.draw_rect(pymupdf.Rect(360.0, 214.5, 506.0, 278.0), color=(1, 1, 1), fill=(1, 1, 1), width=0)
    dest = pymupdf.Rect(clip.x0, clip.y0 + drop, clip.x1, clip.y1 + drop)
    page.show_pdf_page(dest, source, 7, clip=clip)
    for block in source[7].get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                box = pymupdf.Rect(span["bbox"])
                if box.intersects(clip):
                    moved = pymupdf.Rect(box.x0 - 0.5, box.y0 + drop - 0.4, box.x1 + 0.8, box.y1 + drop + 0.4)
                    page.draw_rect(moved, color=(1, 1, 1), fill=(1, 1, 1), width=0)


def build_fixed():
    """Entry point. draw_key is unused; draw_key_fixed is the layout that is painted."""
    regular, bold = font_paths()
    measure = pymupdf.Font(fontfile=str(regular))
    source = next(ROOT.glob("_ICLR_2027__Manifold*.pdf"))
    doc = pymupdf.open(source)
    page = doc[7]
    replace_sockets(doc, page)
    page.wrap_contents()

    titled, callouts, key_spans, curve_labels, axis_labels, rotated_labels = [], [], [], [], [], []
    plan_legend, g_labels = [], []
    event_captions = (
        (pymupdf.Rect(365.4, 95.2, 409.2, 124.4), "Initial  0.00 s"),
        (pymupdf.Rect(410.9, 95.2, 454.6, 124.4), "Contact  0.20 s"),
        (pymupdf.Rect(456.3, 95.2, 500.1, 124.4), "Complete  0.94 s"),
    )
    legend_boxes = {
        "c": pymupdf.Rect(130.4, 264.0, 222.5, 276.2),
        "d": pymupdf.Rect(279.7, 266.8, 339.3, 275.3),
    }
    legend = {name: legend_spans(page, rect) for name, rect in legend_boxes.items()}
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            horizontal = abs(line.get("dir", (1, 0))[0]) > 0.5
            for span in line["spans"]:
                text = span["text"]
                box = pymupdf.Rect(span["bbox"])
                if text in TITLES:
                    titled.append(span)
                elif text in CALLOUTS and 100 < box.x0 < 360:
                    callouts.append(span)
                elif text in CALLOUTS and box.x0 > 360:
                    curve_labels.append(span)
                elif text == "raw peak" and box.x0 < 360:
                    curve_labels.append(span)
                elif 186 < box.y0 < 197 and 100 < box.x0 < 365:
                    key_spans.append(span)
                elif text in {caption for _, caption in event_captions} and box.x0 > 360:
                    curve_labels.append(span)
                elif horizontal and 200 < box.y0 < 264 and 108 < box.x0 < 365 and not text.startswith("("):
                    axis_labels.append(span)
                elif (not horizontal) and 108 < box.x0 < 360 and 190 < box.y1 < 270:
                    rotated_labels.append(span)
                elif 390 < box.x0 < 490 and 258 < box.y0 < 274 and text in {
                    "MGA", "w/o prior", "ISSA", "DIAL", "Complete"}:
                    plan_legend.append(span)
                elif text == "Limit" and box.x0 > 450 and 140 < box.y0 < 160:
                    curve_labels.append(span)
                elif horizontal and box.x0 > 360 and 215 < box.y0 < 258 and not text.startswith("("):
                    g_labels.append(span)

    # The raw-peak words are covered by the new frame instead of redacted,
    # because their boxes touch the axis tick labels.
    for span in titled + callouts + curve_labels + axis_labels + rotated_labels + plan_legend + g_labels:
        redact(page, pymupdf.Rect(span["bbox"]))
    # The force-scale label sits against the 99.0% coverage text. Keep the
    # redaction below that label so the coverage number stays intact.
    for span in key_spans:
        rect = pymupdf.Rect(span["bbox"])
        if rect.x0 < 300 and rect.x1 > 270:
            rect.y0 = max(rect.y0, 189.8)
        redact(page, rect)
    for group in legend.values():
        for span in group:
            redact(page, pymupdf.Rect(span["bbox"]))
    page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE)
    page.insert_font(fontname="body", fontfile=str(regular))
    page.insert_font(fontname="bodyb", fontfile=str(bold))

    f_title = None
    g_title = None
    for span in titled:
        origin = span["origin"]
        text = TITLES[span["text"]]
        if text.startswith("(f)"):
            # Drawn after the wrench panel, so the white cover does not erase the second line.
            f_title = (origin[0], origin[1] - F_SHIFT)
            continue
        if text.startswith("(b)"):
            limit = 358.0
            width = pymupdf.Font(fontfile=str(bold)).text_length(text, fontsize=TITLE)
            size = TITLE if origin[0] + width <= limit else TITLE * (limit - origin[0]) / width
            put(page, origin, text, size, "bodyb")
            continue
        if text.startswith("(d)"):
            c_text = TITLES["(c) Hybrid response (3-pt mean)"]
            c_origin = next(item["origin"] for item in titled if item["text"].startswith("(c)"))
            c_width = pymupdf.Font(fontfile=str(bold)).text_length(c_text, fontsize=TITLE)
            origin = (max(origin[0], c_origin[0] + c_width + 8.0), origin[1])
            width = pymupdf.Font(fontfile=str(bold)).text_length(text, fontsize=TITLE)
            limit = 360.0
            size = TITLE if origin[0] + width <= limit else TITLE * (limit - origin[0]) / width
            put(page, origin, text, size, "bodyb")
            continue
        if text.startswith("(e)"):
            width = pymupdf.Font(fontfile=str(bold)).text_length(text, fontsize=TITLE)
            limit = 502.0
            size = TITLE if origin[0] + width <= limit else TITLE * (limit - origin[0]) / width
            put(page, (origin[0], origin[1] + E_DROP), text, size, "bodyb")
            continue
        if text.startswith("(g)"):
            g_title = (origin[0], origin[1] + G_DROP)
            continue
        put(page, origin, text, TITLE, "bodyb")
    for span in callouts:
        put(page, span["origin"], CALLOUTS[span["text"]], LABEL, "body", color_of(span))
    for span in axis_labels:
        put(page, span["origin"], span["text"], LABEL, "body", color_of(span))
    for span in rotated_labels:
        put_rotated_axis(page, span)
    d_legend = []
    for name, group in legend.items():
        if name == "c":
            continue
        if name == "d":
            d_legend = group
            continue
        for span in group:
            put(page, span["origin"], span["text"], LABEL, "body", color_of(span))
    frames = place_event_frames(page, doc)
    labels = tuple(text for _, text in event_captions)
    paint_event_captions(page, measure, list(zip(frames, labels)))
    draw_key_fixed(page, measure)
    draw_wrench(page, measure)
    if f_title is not None:
        line1, line2 = TITLES["(f) Executed wrench response"].split("\n")
        put(page, f_title, line1, TITLE, "bodyb")
        put(page, (f_title[0], f_title[1] + 7.1), line2, TITLE, "bodyb")
    c_right = draw_response_legend(page, measure)
    draw_clear_response(page, callouts)
    shift_plan_chart(page, G_DROP)
    if g_title is not None:
        g_text = TITLES["(g) Logged plan selection"]
        width = pymupdf.Font(fontfile=str(bold)).text_length(g_text, fontsize=TITLE)
        size = TITLE if g_title[0] + width <= 502.0 else TITLE * (502.0 - g_title[0]) / width
        put(page, g_title, g_text, size, "bodyb")
    for span in g_labels:
        text = "Time (s)" if span["text"] == "Physical time (s)" else span["text"]
        origin = span["origin"]
        if span["text"] == "Physical time (s)":
            old = pymupdf.Rect(span["bbox"])
            width = measure.text_length(text, fontsize=LABEL)
            origin = ((old.x0 + old.x1) / 2 - width / 2, origin[1] + 1.2 + G_DROP)
        elif text in {"Prior adopted", "Refined unsafe", "Emergency"}:
            width = measure.text_length(text, fontsize=LABEL)
            origin = (399.6 - width, origin[1] + G_DROP)
        else:
            origin = (origin[0], origin[1] + G_DROP)
        put(page, origin, text, LABEL, "body", color_of(span))
    place_d_legend(page, d_legend, c_right)

    OUT.mkdir(parents=True, exist_ok=True)
    revised = pymupdf.open()
    revised.insert_pdf(doc, from_page=7, to_page=7)
    pdf_path = OUT / "figure3_revised.pdf"
    revised.save(pdf_path)
    figure = pymupdf.Rect(108, 75, 504, 282)
    pix = revised[0].get_pixmap(matrix=pymupdf.Matrix(4, 4), clip=figure, alpha=False)
    png_path = OUT / "figure3_revised.png"
    pix.save(png_path)
    # The gutter between (d) and (e) is empty from x=360 to x=362.
    slices = (
        (OUT / "figure3_abcd.pdf", pymupdf.Rect(figure.x0, figure.y0, 361.5, figure.y1)),
        (OUT / "figure3_efg.pdf", pymupdf.Rect(360.0, figure.y0, figure.x1, figure.y1)),
    )
    for path, clip in slices:
        piece = pymupdf.open()
        dest = piece.new_page(width=clip.width, height=clip.height)
        dest.show_pdf_page(dest.rect, revised, 0, clip=clip)
        piece.save(path)
        piece.close()
        print(path)
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    build_fixed()
