#!/usr/bin/env python3
"""Build a silent, text-free paper and hardware gallery from native footage.

Install Pillow, numpy and imageio-ffmpeg. The checked-in v2 inputs are sufficient
to regenerate the animation; external paper directories are never required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs/assets"
SOURCE = ASSETS / "showcase_sources/v2"
SIZE = (1680, 1240)
VIDEO_SIZE = (1344, 992)
FPS = 12
SECONDS = 24
BG = "#0B1018"
PANEL = "#141D27"

# A camera travels across one continuous gallery. Wider gutters distinguish
# paper groups without putting labels, legends or graphics over the footage.
CAMERA = [
    (0.0, 336, 162, 640),
    (1.8, 336, 162, 640),
    (3.5, 1172, 242, 1000),
    (5.8, 1172, 242, 1000),
    (7.1, 336, 574, 700),
    (9.2, 336, 574, 700),
    (10.5, 1172, 654, 1000),
    (12.8, 1172, 654, 1000),
    (14.1, 450, 1032, 880),
    (16.4, 1230, 1032, 880),
    (18.4, 840, 620, 1680),
    (22.2, 840, 620, 1680),
    (24.0, 336, 162, 640),
]


def ffmpeg_executable() -> str:
    binary = shutil.which("ffmpeg")
    if binary:
        return binary
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError("Install FFmpeg or imageio-ffmpeg.") from exc


class Clip:
    def __init__(self, metadata: dict, binary: str):
        self.metadata = metadata
        path = SOURCE / metadata["filename"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError(f"Source checksum mismatch: {path.name}")
        self.fps = 8
        limit = 480 if metadata["group"] == "hardware" else 384
        ratio = min(1.0, limit / max(metadata["width"], metadata["height"]))
        self.width = round(metadata["width"] * ratio)
        self.height = round(metadata["height"] * ratio)
        data = subprocess.run([
            binary, "-v", "error", "-i", str(path), "-an", "-vf",
            f"fps={self.fps},scale={self.width}:{self.height}:flags=lanczos",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ], check=True, capture_output=True).stdout
        self.frames = np.frombuffer(data, dtype=np.uint8).reshape(
            -1, self.height, self.width, 3
        )

    def at(self, seconds: float) -> Image.Image:
        index = int(seconds * self.fps) % len(self.frames)
        return Image.fromarray(self.frames[index])


def grid(ids: list[str], box: tuple[int, int, int, int], columns: int,
         gap: int = 8) -> list[tuple[str, tuple[int, int, int, int]]]:
    x, y, width, height = box
    rows = (len(ids) + columns - 1) // columns
    tile_w = (width - gap * (columns - 1)) / columns
    tile_h = (height - gap * (rows - 1)) / rows
    return [(name, (round(x + i % columns * (tile_w + gap)),
                    round(y + i // columns * (tile_h + gap)),
                    round(tile_w), round(tile_h)))
            for i, name in enumerate(ids)]


def layout(manifest: dict) -> tuple[Image.Image, list]:
    groups = {group["id"]: group["clips"] for group in manifest["groups"]}
    metadata = {clip["id"]: clip for clip in manifest["clips"]}
    base = Image.new("RGB", SIZE, BG)
    draw = ImageDraw.Draw(base)
    boxes = [(16, 16, 640, 292), (680, 16, 984, 452),
             (16, 332, 640, 484), (680, 492, 984, 324),
             (16, 840, 1648, 384)]
    for x, y, width, height in boxes:
        draw.rounded_rectangle((x, y, x + width - 1, y + height - 1),
                               radius=7, fill=PANEL)
    tiles = grid(groups["mdoc"], (22, 22, 628, 280), 4)
    # Each scene pairs the diffusion process with its 20 native trajectory
    # candidates. Keep the execution view beside these pairs, spanning both rows.
    x = 686
    for process, candidates, width in (
        ("mdcoas_2d_l6", "mdcoas_2d_l6_candidates", 220),
        ("mdcoas_2d_l10", "mdcoas_2d_l10_candidates", 220),
        ("mdcoas_7dof_diffusion", "mdcoas_7dof_candidates", 224),
    ):
        tiles.append((process, (x, 22, width, 216)))
        tiles.append((candidates, (x, 246, width, 216)))
        x += width + 8
    tiles.append(("mdcoas_7dof_execution", (x, 22, 284, 440)))
    tiles += grid(groups["twogo"], (22, 338, 628, 472), 2)
    tiles += grid(groups["mga"], (686, 498, 972, 312), 6)
    # Use the complete clean crops, preserving the portrait camera geometry.
    aspect = [metadata[name]["width"] / metadata[name]["height"]
              for name in groups["hardware"]]
    height = min(372, (1636 - 8 * (len(aspect) - 1)) / sum(aspect))
    widths = [round(height * value) for value in aspect]
    x = 22 + (1636 - sum(widths) - 8 * (len(widths) - 1)) // 2
    y = 846 + round((372 - height) / 2)
    for name, width in zip(groups["hardware"], widths):
        tiles.append((name, (x, y, width, round(height))))
        x += width + 8
    return base, tiles


def gallery(clips: dict, tiles: list, base: Image.Image, seconds: float) -> Image.Image:
    canvas = base.copy()
    for name, (x, y, width, height) in tiles:
        view = ImageOps.contain(clips[name].at(seconds), (width, height),
                                Image.Resampling.LANCZOS)
        mask = Image.new("L", view.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle(
            (0, 0, view.width - 1, view.height - 1), radius=3, fill=255
        )
        canvas.paste(view, (x + (width - view.width) // 2,
                            y + (height - view.height) // 2), mask)
    return canvas


def viewport(seconds: float) -> tuple[float, float, float, float]:
    for start, end in zip(CAMERA, CAMERA[1:]):
        if start[0] <= seconds <= end[0]:
            phase = (seconds - start[0]) / (end[0] - start[0])
            weight = phase**3 * (phase * (phase * 6 - 15) + 10)
            cx, cy, width = [a + (b - a) * weight
                             for a, b in zip(start[1:], end[1:])]
            height = width * SIZE[1] / SIZE[0]
            left = min(max(cx - width / 2, 0), SIZE[0] - width)
            top = min(max(cy - height / 2, 0), SIZE[1] - height)
            return left, top, left + width, top + height
    raise ValueError(f"No camera keyframe for {seconds}")


def camera(canvas: Image.Image, seconds: float) -> Image.Image:
    return canvas.transform(VIDEO_SIZE, Image.Transform.EXTENT, viewport(seconds),
                            Image.Resampling.BICUBIC)


def build(output: Path, gif_width: int, gif_fps: int, preview_only: bool) -> None:
    binary = ffmpeg_executable()
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    clips = {item["id"]: Clip(item, binary) for item in manifest["clips"]}
    base, tiles = layout(manifest)
    output.mkdir(parents=True, exist_ok=True)
    # Show complete candidate horizons in the static gallery preview.
    poster = gallery(clips, tiles, base, SECONDS - 1 / FPS)
    poster.resize(VIDEO_SIZE, Image.Resampling.LANCZOS).save(
        output / "showcase-poster.png", optimize=True
    )
    if preview_only:
        # A contact sheet for checking the camera journey, never embedded in
        # the shipped animation. It contains no added text either.
        thumb_size = (VIDEO_SIZE[0] // 2, VIDEO_SIZE[1] // 2)
        sheet = Image.new("RGB", (thumb_size[0] * 2, thumb_size[1] * 3), BG)
        for index, seconds in enumerate([1, 4.5, 8, 11.5, 14.5, 20]):
            frame = camera(gallery(clips, tiles, base, seconds), seconds)
            sheet.paste(frame.resize(thumb_size, Image.Resampling.LANCZOS),
                        (index % 2 * thumb_size[0], index // 2 * thumb_size[1]))
        sheet.save(output / "camera-preview.jpg", quality=92)
        print(f"Preview: {output}", flush=True)
        return
    video = output / "showcase.mp4"
    process = subprocess.Popen([
        binary, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{VIDEO_SIZE[0]}x{VIDEO_SIZE[1]}", "-r", str(FPS), "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "slow", "-crf", "19", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(video),
    ], stdin=subprocess.PIPE)
    try:
        for index in range(FPS * SECONDS):
            seconds = index / FPS
            frame = camera(gallery(clips, tiles, base, seconds), seconds)
            process.stdin.write(frame.tobytes())
            if index % (FPS * 4) == 0:
                print(f"Composed {index // FPS}/{SECONDS} seconds", flush=True)
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("FFmpeg video export failed.")
    subprocess.run([
        binary, "-y", "-loglevel", "error", "-i", str(video), "-filter_complex",
        f"fps={gif_fps},scale={gif_width}:-1:flags=lanczos,split[a][b];"
        "[a]palettegen=stats_mode=full[p];"
        "[b][p]paletteuse=dither=bayer:bayer_scale=3",
        "-loop", "0", str(output / "showcase.gif"),
    ], check=True)
    for name in ("showcase.gif", "showcase.mp4", "showcase-poster.png"):
        path = output / name
        print(f"{path.name}: {path.stat().st_size / 1024**2:.2f} MiB", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ASSETS)
    parser.add_argument("--gif-width", type=int, default=896)
    parser.add_argument("--gif-fps", type=int, default=6)
    parser.add_argument("--preview-only", action="store_true")
    args = parser.parse_args()
    build(args.output, args.gif_width, args.gif_fps, args.preview_only)
