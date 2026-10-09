#!/usr/bin/env python3
"""Prepare the text-free, self-contained inputs for the README showcase.

Original paper media are read without modification. The exported MP4 clips and
their provenance manifest are sufficient to rebuild the public showcase; the
original research folders are needed only when preparing these inputs again.

Requires FFmpeg and Pillow, or ``pip install imageio-ffmpeg pillow``. Example::

    python scripts/visualizations/prepare_readme_media.py \
        --mdoc-root /path/to/mdoc \
        --mga-root /path/to/additional_47668 \
        --ral-root /path/to/ral_submission \
        --d3il-gif /path/to/d3il_env.gif

Cropping removes burned-in annotations only. Colors and camera geometry are
preserved. Time is rescaled to one complete source segment per exported loop.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


@dataclass(frozen=True)
class ClipSpec:
    id: str
    group: str
    root: str
    path: str
    crop: tuple[int, int, int, int] | None = None
    start: float = 0.0
    end: float | None = None
    duration: float = 6.0
    max_edge: int = 512


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_media(ffmpeg: str, path: Path) -> dict:
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    duration = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", result.stderr)
    video = next((line for line in result.stderr.splitlines() if "Video:" in line), "")
    shape = re.search(r"\b(\d{2,5})x(\d{2,5})\b", video)
    if not shape:
        raise RuntimeError(f"Could not inspect {path.name}: {result.stderr}")
    if duration:
        hours, minutes, seconds = duration.groups()
        duration_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    else:
        # The supplied D3IL demo has a .gif filename but is an APNG. FFmpeg
        # reports its duration as N/A; Pillow reads the actual frame timings.
        from PIL import Image
        with Image.open(path) as image:
            duration_seconds = 0.0
            for frame in range(getattr(image, "n_frames", 1)):
                image.seek(frame)
                duration_seconds += float(image.info.get("duration", 100.0)) / 1000
    return {
        "width": int(shape.group(1)),
        "height": int(shape.group(2)),
        "duration": duration_seconds,
    }


def clip_specs(roots: dict[str, Path]) -> list[ClipSpec]:
    clips: list[ClipSpec] = []
    for name in ("conveyor", "narrow", "random", "tennis"):
        clips.append(ClipSpec(
            f"mdoc_single_{name}", "mdoc", "mdoc",
            f"assets/vis/gifs/MDOC/{name}_MDOC.gif", (0, 24, 480, 480),
        ))
    for name, agents in (("conveyor_boundary", 10), ("dropregion_boundary", 10),
                         ("empty_large", 15), ("random", 10)):
        clips.append(ClipSpec(
            f"mdoc_cbs_{name}", "mdoc", "mdoc",
            f"assets/vis/gifs/MDOC-CBS/{name}_MDOC-CBS_{agents}.gif",
            (80, 0, 560, 480),
        ))
    for level in (6, 10):
        seed = 8 if level == 10 else 0  # L10 seed 8 is the paper figure.
        clips.append(ClipSpec(
            f"mdcoas_2d_l{level}", "mdcoas", "project",
            f"results/single2d/mdcoas/level_{level}/seed_{seed}/diffusion_steps/diffusion_steps.gif",
            (14, 44, 710, 744),
        ))
        clips.append(ClipSpec(
            f"mdcoas_2d_l{level}_candidates", "mdcoas", "project",
            f"results/single2d/mdcoas/level_{level}/seed_{seed}/trajectory/trajectory_modes.gif",
            (10, 10, 627, 627),
        ))
    clips.append(ClipSpec(
        "mdcoas_7dof_diffusion", "mdcoas", "project",
        "results/d3il_avoiding/mdcoas/level_1/seed_0/diffusion_steps/diffusion_steps.gif",
        (14, 44, 610, 744),
    ))
    clips.append(ClipSpec(
        "mdcoas_7dof_candidates", "mdcoas", "project",
        "results/d3il_avoiding/mdcoas/level_1/seed_0/trajectory/trajectory_modes_plan.gif",
        (10, 10, 539, 627),
    ))
    clips.append(ClipSpec(
        "mdcoas_7dof_execution", "mdcoas", "mdcoas_demo", "d3il_env.gif",
    ))

    # Prefer the original, annotation-free simulation renders over the smaller
    # supplementary contact sheets. The fallback crops retain the same scenes.
    for level, suite, fallback_crop in (
        (1, "s1", (0, 28, 380, 194)),
        (5, "s2", (380, 28, 760, 194)),
    ):
        source = (
            "results/quadruped/stepping_stones_2d/deploy/governed/twogo/"
            f"level_{level}/seed_2/governed/trajectory_mujoco.gif"
        )
        if (roots["project"] / source).is_file():
            clips.append(ClipSpec(f"twogo_go2_{suite}", "twogo", "project", source))
        else:
            clips.append(ClipSpec(
                f"twogo_go2_{suite}", "twogo", "ral_submission",
                "gifs/2go_stepping_S1_S2_go2_execution.gif", fallback_crop,
            ))
    corridor_crops = {
        "a": (0, 28, 240, 178), "b": (240, 28, 480, 178),
        "c": (0, 206, 240, 356), "d": (240, 206, 480, 356),
    }
    for zone, fallback_crop in corridor_crops.items():
        source = (
            "results/humanoid/corridor_2d/deploy/governed/"
            f"twogo_zone_{zone}/level_1/seed_2/trajectory_mujoco.gif"
        )
        if (roots["project"] / source).is_file():
            clips.append(ClipSpec(f"twogo_g1_zone_{zone}", "twogo", "project", source))
        else:
            clips.append(ClipSpec(
                f"twogo_g1_zone_{zone}", "twogo", "ral_submission",
                "gifs/2go_corridor_zoneA-D_g1_execution.gif", fallback_crop,
            ))

    mga_names = (
        "surface_rigid_plane", "surface_rigid_convex", "surface_rigid_cylinder",
        "surface_rigid_bumpy", "surface_soft_plane", "surface_soft_convex",
        "surface_soft_cylinder", "surface_soft_bumpy", "surface_soft_unseen",
        "surface_hybrid_center_hard", "surface_hybrid_center_soft", "surface_hybrid_stripes",
        "peg_id_wide", "peg_ood_pose", "peg_ood_sensing",
        "humanoid_force_regulation", "humanoid_fixed_stance_push", "humanoid_unjamming",
    )
    for name in mga_names:
        crop = (0, 0, 810, 540) if name.startswith("humanoid") else (0, 0, 870, 580)
        clips.append(ClipSpec(
            f"mga_{name}", "mga", "mga_supplement", f"gifs/mga_{name}.gif", crop,
        ))
    clips.append(ClipSpec(
        "hardware_twogo_g1", "hardware", "ral_submission", "video/2go.mp4",
        (400, 80, 1310, 930), start=92, end=102, duration=8, max_edge=640,
    ))
    for name, crop in (
        ("mga_surface_foam", (80, 160, 710, 950)),
        ("mga_peg_insert", (1560, 160, 2160, 940)),
        ("mga_surface_curved", (350, 1240, 1080, 2010)),
        ("mga_surface_rigid", (1100, 1240, 1770, 2020)),
    ):
        clips.append(ClipSpec(
            f"hardware_{name}", "hardware", "mga_supplement", "videos/deployment.mp4",
            crop, start=12, end=24, duration=8, max_edge=640,
        ))
    assert len(clips) == 44
    return clips


def prepare_clip(spec: ClipSpec, roots: dict[str, Path], output: Path, ffmpeg: str) -> dict:
    source = roots[spec.root] / spec.path
    if not source.is_file():
        raise FileNotFoundError(f"Missing {spec.root}/{spec.path}")
    source_info = inspect_media(ffmpeg, source)
    crop = spec.crop or (0, 0, source_info["width"], source_info["height"])
    x0, y0, x1, y1 = crop
    if not (0 <= x0 < x1 <= source_info["width"] and 0 <= y0 < y1 <= source_info["height"]):
        raise ValueError(f"Crop outside source for {spec.id}: {crop}")
    end = source_info["duration"] if spec.end is None else spec.end
    if not (0 <= spec.start < end <= source_info["duration"] + 0.01):
        raise ValueError(f"Invalid time segment for {spec.id}")
    segment_duration = end - spec.start
    width, height = x1 - x0, y1 - y0
    scale = min(1.0, spec.max_edge / max(width, height))
    width = max(2, round(width * scale / 2) * 2)
    height = max(2, round(height * scale / 2) * 2)
    destination = output / f"{spec.id}.mp4"
    filters = (
        f"trim=duration={segment_duration:.8f},"
        f"setpts={spec.duration / segment_duration:.12f}*(PTS-STARTPTS),"
        f"crop={x1-x0}:{y1-y0}:{x0}:{y0},"
        f"scale={width}:{height}:flags=lanczos,fps=12,"
        # setpts rescales timestamps but not a GIF's final-frame hold. Pad the
        # last decoded frame, then cap output to the exact normalized duration.
        f"tpad=stop_mode=clone:stop_duration={spec.duration},format=yuv420p"
    )
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-threads", "1"]
    if source.suffix.lower() == ".gif":
        command += ["-ignore_loop", "1"]
    if spec.start:
        command += ["-ss", str(spec.start)]
    command += [
        "-i", str(source), "-vf", filters, "-t", str(spec.duration),
        "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "20",
        "-threads", "1", "-movflags", "+faststart", "-map_metadata", "-1",
        "-y", str(destination),
    ]
    subprocess.run(command, capture_output=True, text=True, check=True)
    actual = inspect_media(ffmpeg, destination)
    if abs(actual["duration"] - spec.duration) > 0.09:
        raise RuntimeError(f"Unexpected output duration for {spec.id}: {actual['duration']}")
    return {
        "id": spec.id, "group": spec.group, "filename": destination.name,
        "width": actual["width"], "height": actual["height"],
        "shape": [actual["width"], actual["height"]],
        "duration": actual["duration"], "normalized_duration": spec.duration,
        "fps": 12, "codec": "h264", "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "origin": {
            "root": spec.root, "path": spec.path, "sha256": sha256(source),
            "source_width": source_info["width"], "source_height": source_info["height"],
            "source_duration": source_info["duration"], "crop": list(crop),
            "start": spec.start, "end": end,
        },
    }


def main() -> None:
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project)
    parser.add_argument("--mdoc-root", type=Path, required=True)
    parser.add_argument("--mga-root", type=Path, required=True)
    parser.add_argument("--ral-root", type=Path, required=True)
    parser.add_argument("--d3il-gif", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=project / "docs/assets/showcase_sources/v2")
    parser.add_argument("--ffmpeg")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    ffmpeg = args.ffmpeg or shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError as exc:
            raise SystemExit("Install FFmpeg or `pip install imageio-ffmpeg`.") from exc
    roots = {
        "project": args.project_root.resolve(), "mdoc": args.mdoc_root.resolve(),
        "mga_supplement": args.mga_root.resolve(), "ral_submission": args.ral_root.resolve(),
        "mdcoas_demo": args.d3il_gif.resolve().parent,
    }
    if args.d3il_gif.name != "d3il_env.gif":
        raise SystemExit("The supplied paper demo must be named d3il_env.gif.")
    specs = clip_specs(roots)
    args.output.mkdir(parents=True, exist_ok=True)
    completed: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {pool.submit(prepare_clip, spec, roots, args.output, ffmpeg): spec for spec in specs}
        for future in as_completed(futures):
            spec = futures[future]
            try:
                completed[spec.id] = future.result()
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f"FFmpeg failed for {spec.id}: {exc.stderr}") from exc
            print(f"[{len(completed):02d}/{len(specs)}] {spec.id}", flush=True)
    group_names = (("mdoc", "MDOC"), ("mdcoas", "MD-COAS"), ("twogo", "2GO"),
                   ("mga", "MGA"), ("hardware", "Hardware"))
    manifest = {
        "schema_version": 2, "fps": 12,
        "description": "Original paper demonstrations, cropped to remove annotations; no added text.",
        "groups": [{"id": key, "label": label, "clips": [spec.id for spec in specs if spec.group == key]}
                   for key, label in group_names],
        "clips": [completed[spec.id] for spec in specs],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    total = sum(clip["bytes"] for clip in manifest["clips"])
    print(f"Prepared {len(specs)} clips, {total / 1024 / 1024:.2f} MiB, at {args.output}")


if __name__ == "__main__":
    main()
