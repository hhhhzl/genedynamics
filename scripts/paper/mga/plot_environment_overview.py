#!/usr/bin/env python3
"""Compose an environment overview, not an experimental-results figure.

Each complete environment panel fits inside the original 3:2 canvas. Detail
insets occupy the right side (plus an internal shape strip for scanning), not
an additional row outside that canvas. Hero renders are scaled uniformly.
Contact and peg insets are schematic: compliance is not mesh deformation,
hybrid colors are task-chart maps, and the Pose-OOD offset is magnified.

Run after render_env_humanoid.py with Python + numpy + matplotlib + Pillow.
No manuscript files or simulation data are modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, ListedColormap
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle
from matplotlib.transforms import Affine2D
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output/env_overview_v4"
INK = "#3A4655"
MUTED = "#6E7984"
RULE = "#DDE3E6"
TEAL = "#1F7A78"
TEAL_FILL = "#D6EDE8"
INDIGO = "#4B2E83"
PURPLE_FILL = "#E3DDEF"
ORANGE = "#D9660E"
LIME = "#5C8F2E"
PAPER = "#FFFFFF"


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 16, "text.color": INK, "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": PAPER, "figure.facecolor": PAPER,
    })


def axes(fig, rect):
    ax = fig.add_axes(rect)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    return ax


def arrow(ax, start, end, color=ORANGE, scale=15, lw=1.8, **kwargs):
    p = FancyArrowPatch(start, end, arrowstyle="-|>", color=color,
                        mutation_scale=scale, linewidth=lw, **kwargs)
    ax.add_patch(p)
    return p


def image_panel(fig, rect, path):
    image = Image.open(path).convert("RGB")
    ratio = rect[2] * fig.get_figwidth() / (rect[3] * fig.get_figheight())
    if abs(image.width / image.height - ratio) > 0.015:
        raise ValueError(f"Image aspect ratio would change: {path}")
    ax = fig.add_axes(rect)
    ax.imshow(image, interpolation="lanczos")
    ax.set_axis_off()
    return ax


def fit_image(fig, rect, path):
    """Place a photo inside rect, centered, without changing its aspect ratio."""
    image = Image.open(path).convert("RGB")
    fig_w, fig_h = fig.get_figwidth(), fig.get_figheight()
    box_aspect = (rect[2] * fig_w) / (rect[3] * fig_h)
    image_aspect = image.width / image.height
    if image_aspect > box_aspect:
        width = rect[2]
        height = width * fig_w / (image_aspect * fig_h)
    else:
        height = rect[3]
        width = height * fig_h * image_aspect / fig_w
    fitted = [rect[0] + (rect[2] - width) / 2, rect[1] + (rect[3] - height) / 2, width, height]
    return image_panel(fig, fitted, path)


def spring(ax, x, y0, y1, color, width=0.035):
    yy = np.linspace(y0, y1, 11)
    xx = x + np.array([0, 0, -1, 1, -1, 1, -1, 1, -1, 0, 0]) * width
    ax.plot(xx, yy, color=color, lw=1.25, solid_capstyle="round")


def material_card(ax, kind):
    """Identical nominal plane, different contact response, not a deformed mesh."""
    # Shared label and shared icon top, so the three gaps match.
    label_top, icon_top = .97, .72
    ax.text(.5, label_top, kind, ha="center", va="top", fontsize=15, weight="bold")
    if kind == "Hybrid":
        # Chart-map diagrams, deliberately not painted onto a world-space mesh.
        maps = [np.tile([0, 1, 0, 1, 0, 1], (6, 1)),
                np.pad(np.ones((2, 2)), 2),
                1 - np.pad(np.ones((2, 2)), 2)]
        cmap = ListedColormap([TEAL_FILL, INDIGO])
        map_h = .36
        map_bottom = icon_top - .03 - map_h
        for i, data in enumerate(maps):
            left = .03 + i * .33
            top = icon_top - .03
            ax.imshow(data, extent=(left, left + .28, map_bottom, top),
                      cmap=cmap, vmin=0, vmax=1, interpolation="nearest", aspect="auto")
            ax.add_patch(Rectangle((left, map_bottom), .28, map_h, fill=False,
                                   edgecolor=RULE, lw=.8))
        return
    hard = kind == "Rigid"
    color, fill = (INDIGO, PURPLE_FILL) if hard else (TEAL, TEAL_FILL)
    # Same probe top. Soft shows penetration by raising the plane into the probe.
    plate_y = .29 if hard else .36
    ax.add_patch(Rectangle((.09, plate_y), .82, .15, color=fill, ec=color, lw=1.0))
    ax.plot([.09, .91], [plate_y + .15, plate_y + .15], color=color, lw=2.0)
    ax.plot([.09, .91], [.08, .08], color=MUTED, lw=.8)
    for x in [.25, .5, .75]:
        spring(ax, x, .10, plate_y, color, width=.018 if hard else .040)
    cy = icon_top - .275
    ax.add_patch(Rectangle((.465, cy + .015), .07, .26, color=INK))
    ax.add_patch(Circle((.5, cy), .071, fc=ORANGE, ec="white", lw=.8))
    if not hard:
        ax.plot([.35, .65], [plate_y + .15, plate_y + .15], color=TEAL, lw=1.0, ls=(0, (2, 2)))
    arrow(ax, (.70, icon_top), (.70, icon_top - .26), scale=13, lw=1.4)


def shape_card(fig, rect, name):
    """Representative shape-family illustration, not measured terrain."""
    ax = fig.add_axes(rect, projection="3d")
    a = np.linspace(-1, 1, 32)
    x, y = np.meshgrid(a, a)
    if name == "Plane":
        z = np.zeros_like(x)
    elif name == "Cylinder":
        z = .55 * np.sqrt(np.maximum(0, 1 - .90 * x**2))
    elif name == "Convex":
        z = .6 * (1 - .7*x*x) * (1 - .7*y*y)
    elif name == "Bumpy":
        z = .14 * np.cos(1.4*np.pi*x) * np.sin(1.4*np.pi*y) + .15
    else:
        z = (.40*np.exp(-5*((x+.4)**2+(y-.15)**2))
             + .28*np.exp(-8*((x-.35)**2+(y+.4)**2))
             + .08*np.sin(5*x+3*y) + .13)
    color = TEAL_FILL if name != "Unseen" else PURPLE_FILL
    ax.plot_surface(x, y, z, color=color, linewidth=0, antialiased=True,
                    shade=True, lightsource=LightSource(azdeg=300, altdeg=45))
    for edge in [(0, slice(None)), (-1, slice(None)), (slice(None), 0), (slice(None), -1)]:
        ax.plot(x[edge], y[edge], z[edge], color=TEAL if name != "Unseen" else INDIGO,
                lw=.5, alpha=.8)
    ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-.1, .65))
    ax.set_box_aspect((1, 1, .35))
    ax.view_init(elev=27, azim=-58)
    ax.set_axis_off()
    ax.set_proj_type("ortho")
    return ax


def socket(ax, center=(.5, .60), theta=0, color=INK, nominal=False):
    """Top view of the local peg/socket geometry with a fixed nominal peg."""
    cx, cy = center
    trans = Affine2D().rotate_deg_around(cx, cy, theta) + ax.transData
    if nominal:
        ax.add_patch(Rectangle((cx-.25, cy-.20), .50, .40, fill=False,
                               edgecolor=MUTED, lw=1.3, ls=(0, (3, 2)), transform=trans))
        ax.add_patch(Rectangle((cx-.15, cy-.11), .30, .22, fill=False,
                               edgecolor=MUTED, lw=1.0, ls=(0, (3, 2)), transform=trans))
    else:
        ax.add_patch(Rectangle((cx-.30, cy-.235), .60, .47,
                               facecolor="#E4E8EC", edgecolor=color, lw=1.3, transform=trans))
        ax.add_patch(Rectangle((cx-.15, cy-.11), .30, .22,
                               facecolor=PAPER, edgecolor=color, lw=1.4, transform=trans))


def peg_card(ax, kind):
    ax.text(.5, .98, kind, ha="center", va="top", fontsize=17, weight="bold")
    if kind == "Pose-OOD":
        socket(ax, center=(.56, .58), theta=11, color=ORANGE)
        socket(ax, nominal=True)
    else:
        socket(ax)
    ax.add_patch(Rectangle((.374, .519), .252, .162, facecolor=TEAL_FILL,
                           edgecolor=TEAL, lw=1.6))
    ax.plot([.5], [.60], marker="+", markersize=5, color=TEAL, mew=.8)
    if kind == "ID":
        ax.text(.5, .19, "Nominal", ha="center", fontsize=15.5)
        ax.text(.5, .065, "Unbiased sensing", ha="center", fontsize=13.5, color=MUTED)
    elif kind == "Pose-OOD":
        arrow(ax, (.48, .81), (.76, .76), scale=12, lw=1.4)
        ax.text(.5, .19, "Shift + rotation", ha="center", fontsize=15.5)
        ax.text(.5, .065, "Offset magnified", ha="center", fontsize=13.5, color=MUTED)
    else:
        arrow(ax, (.50, .60), (.63, .80), color=TEAL, scale=12, lw=1.5)
        arrow(ax, (.50, .60), (.80, .73), color=ORANGE, scale=12, lw=1.5)
        ax.text(.5, .19, "Wrench bias", ha="center", fontsize=15.5)
        ax.text(.5, .065, r"$\widehat{\mathbf{w}}=\mathbf{w}+\mathbf{b}$",
                ha="center", fontsize=14, color=MUTED)


def compact_peg(ax, kind):
    """A single local inset; explanatory prose belongs in the figure caption."""
    ax.text(.5, .99, kind, ha="center", va="top", fontsize=15, weight="bold")
    if kind == "Pose-OOD":
        socket(ax, center=(.56, .44), theta=11, color=ORANGE)
        socket(ax, center=(.5, .46), nominal=True)
        arrow(ax, (.48, .77), (.76, .72), scale=12, lw=1.5)
    else:
        socket(ax, center=(.5, .46), color=INDIGO)
    ax.add_patch(Rectangle((.374, .379), .252, .162, facecolor=TEAL_FILL,
                           edgecolor=TEAL, lw=1.5))
    if kind == "Sensing-OOD":
        arrow(ax, (.50, .46), (.63, .73), color=TEAL, scale=12, lw=1.5)
        arrow(ax, (.50, .46), (.81, .63), color=ORANGE, scale=12, lw=1.5)


def recolor_peg_socket(path, destination):
    """Shift only the blue socket block to the inset teal, keeping its shading."""
    image = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    socket = (blue > 150) & ((blue - red) > 85) & (green < 125) & (red < 90)
    teal = np.array([176, 214, 210], np.float32)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    socket_luminance = float(luminance[socket].mean())
    painted = np.clip(teal * (luminance / socket_luminance)[..., None], 0, 255)
    image[socket] = painted[socket]
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.astype(np.uint8)).save(destination)
    return destination


def hero_sources(output):
    """Prefer the manuscript screenshots. Fall back to rasters already extracted here."""
    latex = [ROOT / "latex/latex_mga/figures/envs/scanning.png",
             ROOT / "latex/latex_mga/figures/envs/peg_insert.png"]
    assets = output / "assets"
    if all(path.exists() for path in latex):
        scanning, peg = latex
    else:
        scanning, peg = assets / "source_scanning.png", assets / "source_peg_insert.png"
    return [scanning, peg, assets / "humanoid_hero.png"]


def versioned_name(directory, stem, suffix):
    """Never replace an existing export. scanning.png stays; the next file is scanning_1.png."""
    number = 1
    while (directory / f"{stem}_{number}{suffix}").exists():
        number += 1
    return f"{stem}_{number}{suffix}"


def make_figure(output, dpi):
    style()
    width, margin, gap = 15.3, .006, .012
    cw = (1 - 2*margin - 2*gap) / 3
    # The entire panel, including its insets, is exactly the original 3:2.
    height = width * cw / 1.5
    fig = plt.figure(figsize=(width, height))
    cols = [margin + i*(cw+gap) for i in range(3)]
    humanoid_assets = output / "assets"
    sources = hero_sources(output)
    titles = ["(a) Surface Scanning", "(b) Peg Insertion", "(c) Humanoid Push"]
    # Titles share one baseline. Right insets use the full panel height.
    inset_bottom, inset_top = .012, .988
    band = (inset_top - inset_bottom) / 3
    right_x, right_w, gutter = .72, .265, .70

    def rect(panel, x, y, w, h):
        return [cols[panel]+cw*x, y, cw*w, h]

    def band_y(index, bottom, band):
        return bottom + (2 - index) * band

    def line(panel, xs, ys):
        fig.add_artist(plt.Line2D([cols[panel]+cw*v for v in xs], ys,
                                  transform=fig.transFigure, lw=.8, color=RULE))

    for i, title in enumerate(titles):
        bg = axes(fig, rect(i, 0, 0, 1, 1))
        bg.add_patch(Rectangle((0, 0), 1, 1, color="white", ec=RULE, lw=.8))
        fig.text(cols[i]+cw*.04, .90, title, ha="left", va="center", fontsize=22,
                 weight="bold", zorder=10, color=INK)

    # Scanning: hero stays left of a clear gutter; Rigid/Soft/Hybrid do not touch it.
    scan_band = band
    image_panel(fig, rect(0, .025, .24, .54, .54), sources[0])
    line(0, [gutter, gutter], [inset_bottom, inset_top])
    line(0, [0, gutter], [.20, .20])
    for i, name in enumerate(["Rigid", "Soft", "Hybrid"]):
        y = band_y(i, inset_bottom, scan_band)
        material_card(axes(fig, rect(0, right_x, y, right_w, scan_band)), name)
        if i < 2:
            line(0, [right_x, .99], [y, y])
    for i, name in enumerate(["Plane", "Cylinder", "Convex", "Bumpy", "Unseen"]):
        shape_card(fig, rect(0, i*.136 + .006, .02, .124, .15), name)
        fig.text(cols[0] + cw*(i*.136 + .068), .012, name, ha="center", va="bottom",
                 fontsize=12, color=INK, weight="bold")

    peg_hero = recolor_peg_socket(sources[1], output / "assets" / "source_peg_insert_teal.png")
    other_band = band
    image_panel(fig, rect(1, .02, .20, .58, .58), peg_hero)
    line(1, [.62, .62], [inset_bottom, inset_top])
    peg_x, peg_w = .64, .34
    for i, name in enumerate(["ID", "Pose-OOD", "Sensing-OOD"]):
        y = band_y(i, inset_bottom, other_band)
        compact_peg(axes(fig, rect(1, peg_x, y, peg_w, other_band)), name)
        if i < 2:
            line(1, [peg_x, .99], [y, y])

    image_panel(fig, rect(2, .02, .20, .58, .58), sources[2])
    line(2, [.62, .62], [inset_bottom, inset_top])
    for i, (key, title) in enumerate([
        ("force_regulation", "Force\nRegulation"),
        ("fixed_stance_push", "Fixed-stance\nPush"),
        ("unjamming", "Unjamming"),
    ]):
        y = band_y(i, inset_bottom, other_band)
        label_h = .09
        ax = fit_image(fig, rect(2, peg_x, y, peg_w, other_band - label_h),
                       humanoid_assets / f"humanoid_{key}.png")
        fig.text(cols[2]+cw*(peg_x+peg_w/2), y+other_band-.008, title,
                 ha="center", va="top", fontsize=15, weight="bold", color=INK,
                 linespacing=0.9)
        if key == "force_regulation":
            start, end, curve = (.42, .59), (.60, .59), 0
        elif key == "fixed_stance_push":
            start, end, curve = (.50, .12), (.78, .12), 0
        else:
            start, end, curve = (.55, .71), (.73, .49), -.6
        ax.annotate("", xy=end, xytext=start, xycoords="axes fraction",
                    arrowprops={"arrowstyle": "-|>", "color": ORANGE, "lw": 1.7,
                                "mutation_scale": 14, "connectionstyle": f"arc3,rad={curve}"})
        if i < 2:
            line(2, [peg_x, .99], [y, y])
    return fig, sources, cols, cw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=280)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, sources, cols, cw = make_figure(args.output_dir, args.dpi)
    overview_name = versioned_name(args.output_dir, "mga_environment_overview_v4", ".png")
    target = args.output_dir / overview_name
    fig.savefig(target, dpi=args.dpi)
    fig.savefig(target.with_suffix(".pdf"), dpi=args.dpi)
    from matplotlib.transforms import Bbox
    panel_names = ["scanning", "peg_insert", "push_to_line"]
    written = []
    for x, name in zip(cols, panel_names):
        physical_width = fig.get_figwidth()*cw
        crop = Bbox.from_bounds(x*fig.get_figwidth(), 0, physical_width, fig.get_figheight())
        destination = args.output_dir / versioned_name(args.output_dir, name, ".png")
        fig.savefig(destination, dpi=args.dpi, bbox_inches=crop, pad_inches=0)
        written.append(destination)
    plt.close(fig)
    sizes = {path.name: list(Image.open(path).size) for path in written}
    provenance = {
        "figure_type": "environment overview; no experimental measurements or results plotted",
        "panel_aspect_ratio": "3:2, with all insets inside each original canvas",
        "original_arm_images_preserved": True,
        "humanoid_box": "v1 Brax-reference box material #C7B384, without sRGB-to-linear conversion; other manuscript colors unchanged",
        "arm_image_transform": "Uniform scaling only; original background, camera, materials and colors preserved.",
        "individual_png_sizes": sizes,
        "palette": "Manuscript teal #1F7A78, indigo #4B2E83, orange #D9660E, slate #3A4655 and associated pale fills",
        "sources": [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                    for p in sources],
        "schematics": {
            "scanning_materials": "Rigid/Soft have the same nominal planar geometry. Spring icons and probe penetration depict contact response, not deformation of the rendered mesh.",
            "hybrid": "Three symbolic task-chart stiffness maps; colors are not world-space surface texture. Hybrid is planar only.",
            "shapes": "Representative normalized shape-family illustrations, not reconstructed or measured heightfields.",
            "peg_pose": "Nominal dashed outline and offset/rotated socket. Offset magnified for visibility, not an actual rollout pose.",
            "peg_sensing": "Same geometry as ID; arrows denote true versus biased wrench observations, not physical motion.",
        },
        "references": ["latex/latex_mga/sections/appendix_environments.tex", "latex/latex_mga/sections/exp.tex"],
        "humanoid_arrows": "Schematic task-intent annotations (force / translation / yaw), not trajectories or measured displacements.",
        "manuscript_modified": False,
    }
    target.with_suffix(".json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(target)
    print(target.with_suffix(".pdf"))
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
