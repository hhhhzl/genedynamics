#!/usr/bin/env python3
"""Compose the README motion wall from real project renders.

Requires Pillow and FFmpeg (or imageio-ffmpeg). No simulator, JAX, or model
checkpoint is needed. Input provenance is in docs/assets/showcase_sources/.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs/assets"
WIDTH, HEIGHT = 1440, 914
FPS, SECONDS = 12, 12
BG = "#0B131D"
PANEL = "#111F2C"
LINE = "#2A3B49"
INK = "#F1F5F7"
MUTED = "#9AABB9"
ACCENT = "#8CE4CF"
CARD_W, CARD_H, VIEW_H = 448, 348, 278
MARGIN, GAP, TOP = 30, 18, 120


def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    """Use a local font, with portable macOS/Linux fallbacks."""
    candidates = (
        [
            "/System/Library/Fonts/Menlo.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
        ]
        if mono else
        [
            f"/System/Library/Fonts/Supplemental/Arial{' Bold' if bold else ''}.ttf",
            f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
            f"/usr/share/fonts/truetype/liberation2/LiberationSans-{'Bold' if bold else 'Regular'}.ttf",
        ]
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise RuntimeError("Install Arial, DejaVu Sans, or Liberation Sans to render the showcase.")


def ffmpeg_executable() -> str:
    binary = shutil.which("ffmpeg")
    if binary:
        return binary
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError("Install FFmpeg or `pip install imageio-ffmpeg`.") from exc


class Clip:
    def __init__(self, path: Path, crop: tuple[int, int, int, int] | None = None):
        self.frames = []
        with Image.open(path) as source:
            for index in range(source.n_frames):
                source.seek(index)
                frame = source.convert("RGB")
                if crop:
                    frame = frame.crop(crop)
                self.frames.append(frame)

    def at(self, phase: float, size: tuple[int, int], *, contain: bool = False) -> Image.Image:
        index = min(int(phase * len(self.frames)), len(self.frames) - 1)
        frame = self.frames[index]
        if contain:
            return ImageOps.contain(frame, size, Image.Resampling.LANCZOS)
        return ImageOps.fit(frame, size, Image.Resampling.LANCZOS)


def plot_style(frame: Image.Image) -> Image.Image:
    """Map the source plot's white/gray/blue/red palette into the motion wall.

    Positions, trajectories, obstacles, and goals remain those of the source
    image. This only changes the display palette; it never re-renders motion.
    """
    rgb = frame.convert("RGB")
    # The source plots use a white background, gray geometry, blue trajectories,
    # and red goal/obstacle markers. A lookup table preserves their distinctions.
    pixels = rgb.load()
    for y in range(rgb.height):
        for x in range(rgb.width):
            r, g, b = pixels[x, y]
            if r > 240 and g > 240 and 185 < b < 240:
                pixels[x, y] = (17, 31, 44)
            elif max(r, g, b) - min(r, g, b) < 14:
                weight = (255 - r) / 255
                pixels[x, y] = tuple(round(a + weight * (z - a)) for a, z in zip((17, 31, 44), (125, 149, 165)))
            elif b > r and b > g:
                weight = min(1, (255 - r) / 170)
                pixels[x, y] = tuple(round(a + weight * (z - a)) for a, z in zip((17, 31, 44), (140, 228, 207)))
            elif r > g * 1.25:
                pixels[x, y] = (244, 171, 120)
    return rgb


def layout() -> Image.Image:
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
    d = ImageDraw.Draw(canvas)
    d.text((MARGIN, 25), "GENERATIVEDYNAMICS", font=font(14, mono=True), fill=ACCENT)
    d.text((MARGIN - 1, 49), "From planning to control.", font=font(39, bold=True), fill=INK)
    d.text((1040, 31), "SIX TASKS", font=font(13, mono=True), fill=MUTED)
    d.text((1040, 54), "ONE MODULAR STACK", font=font(17, bold=True), fill=INK)
    # A small trajectory glyph creates a recognizable mark without a decorative
    # logo asset or a synthetic robot image.
    for offset in (0, 10, 20):
        pts = [(1358 + offset // 3, 76), (1364 + offset // 2, 62), (1382, 58 - offset // 2), (1395 + offset // 2, 33)]
        d.line(pts, fill=ACCENT if offset == 10 else LINE, width=2)
    d.ellipse((1395, 27, 1401, 33), fill=ACCENT)

    labels = [
        ("Compliant surface scanning", "MGA  /  GEOMETRY + FORCE"),
        ("Contact-rich peg insertion", "MGA  /  CONTACT CONTROL"),
        ("Obstacle avoidance", "EB-MBD  /  D3IL"),
        ("Multi-modal foothold planning", "2GO  /  QUADRUPED"),
        ("Constrained corridor motion", "2GO  /  HUMANOID"),
        ("Generative trajectory planning", "MBD  /  NON-CONVEX SCENES"),
    ]
    for index, (title, detail) in enumerate(labels):
        col, row = index % 3, index // 3
        x, y = MARGIN + col * (CARD_W + GAP), TOP + row * (CARD_H + GAP)
        d.rounded_rectangle((x, y, x + CARD_W - 1, y + CARD_H - 1), radius=12, fill=PANEL, outline=LINE, width=1)
        d.text((x + 17, y + VIEW_H + 13), title, font=font(22, bold=True), fill=INK)
        d.text((x + 18, y + VIEW_H + 43), detail, font=font(12, mono=True), fill=MUTED)
    d.line((MARGIN, 860, WIDTH - MARGIN, 860), fill=LINE, width=1)
    d.text((MARGIN, 878), "PLAN  /  CONSTRAIN  /  EXECUTE", font=font(12, mono=True), fill=ACCENT)
    d.text((934, 878), "PROJECT RENDERS  ·  PLAYBACK RESCALED", font=font(11, mono=True), fill=MUTED)
    return canvas


def compose(clips: list[Clip], inset: Clip, phase: float, base: Image.Image) -> Image.Image:
    canvas = base.copy()
    for index, clip in enumerate(clips):
        col, row = index % 3, index // 3
        x, y = MARGIN + col * (CARD_W + GAP), TOP + row * (CARD_H + GAP)
        # The contact clips play twice per wall cycle, with a short final hold.
        local_phase = min((phase * 2) % 1 / .86, 1) if index < 2 else phase
        view = clip.at(local_phase, (CARD_W - 2, VIEW_H - 1), contain=index == 5)
        if index == 5:
            view = plot_style(view)
            scene = Image.new("RGB", (CARD_W - 2, VIEW_H - 1), PANEL)
            scene.paste(view, ((scene.width - view.width) // 2, 0))
            view = scene
            # Quiet dashed guide lines fill the wide plot panel without changing
            # the original data extent or implying a different trajectory.
            guide = ImageDraw.Draw(view)
            guide.text((17, 19), "NOISE", font=font(10, mono=True), fill=MUTED)
            guide.text((336, 235), "MOTION", font=font(10, mono=True), fill=ACCENT)
        if index == 2:
            # Both views come from EB-MBD / level_0 / seed_0. The independently
            # sampled source renders show the same run, not synchronized clocks.
            mini = inset.at(phase, (101, 117), contain=True)
            mini = plot_style(mini)
            box = Image.new("RGB", (111, 142), PANEL)
            box.paste(mini, ((111 - mini.width) // 2, 6))
            md = ImageDraw.Draw(box)
            md.text((8, 125), "2D VIEW", font=font(9, mono=True), fill=ACCENT)
            md.rounded_rectangle((0, 0, 110, 141), radius=6, outline=LINE)
            view.paste(box, (view.width - box.width - 12, 12))
        mask = Image.new("L", view.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, view.width, view.height + 12), radius=11, fill=255)
        canvas.paste(view, (x + 1, y + 1), mask)
    return canvas


def build(output: Path, gif_width: int, gif_fps: int) -> None:
    binary = ffmpeg_executable()
    source_dir = ASSETS / "showcase_sources"
    clips = [
        Clip(ASSETS / "mga_surface_scan.gif", (0, 0, 870, 580)),
        Clip(ASSETS / "mga_peg_insert.gif", (0, 0, 870, 580)),
        Clip(source_dir / "d3il_3d.webp"),
        Clip(source_dir / "quadruped.webp"),
        Clip(source_dir / "humanoid_corridor.webp"),
        Clip(source_dir / "planar_diffusion.webp"),
    ]
    inset = Clip(ASSETS / "d3il_avoiding.gif", (17, 18, 531, 609))
    base = layout()
    output.mkdir(parents=True, exist_ok=True)
    video = output / "showcase.mp4"
    process = subprocess.Popen([
        binary, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(video),
    ], stdin=subprocess.PIPE)
    try:
        for index in range(FPS * SECONDS):
            canvas = compose(clips, inset, index / (FPS * SECONDS), base)
            if index == FPS * 5:
                canvas.save(output / "showcase-poster.png", optimize=True)
            process.stdin.write(canvas.tobytes())
            if index % FPS == 0:
                print(f"Composed {index // FPS + 1}/{SECONDS} seconds", flush=True)
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("FFmpeg video export failed.")
    subprocess.run([
        binary, "-y", "-loglevel", "error", "-i", str(video),
        "-filter_complex",
        f"fps={gif_fps},scale={gif_width}:-1:flags=lanczos,split[a][b];"
        "[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3",
        "-loop", "0", str(output / "showcase.gif"),
    ], check=True)
    for name in ("showcase.gif", "showcase.mp4", "showcase-poster.png"):
        path = output / name
        print(f"{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}: {path.stat().st_size / 1024**2:.2f} MiB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ASSETS)
    parser.add_argument("--gif-width", type=int, default=1080)
    parser.add_argument("--gif-fps", type=int, default=10)
    args = parser.parse_args()
    build(args.output, args.gif_width, args.gif_fps)
