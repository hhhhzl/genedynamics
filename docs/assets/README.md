# Demonstration gallery

The homepage animation is one continuous, silent gallery of **41 native paper
and hardware clips**. A camera visits five groups, then pulls back to the whole
wall before returning to the first group. Wider gutters separate the groups.
There are no added titles, captions, logos or labels inside the animation.

| Order | Group | Native footage |
| --- | --- | --- |
| 1 | **MDOC** | Four single-robot scenes and four multi-robot CBS scenes from the original MDOC project. |
| 2 | **MD-COAS** | Two planar diffusion scenes (including the paper’s L10, seed 8 example), 7-DoF avoidance diffusion and the corresponding D3IL execution environment. |
| 3 | **2GO** | Two quadruped stepping-stone scenes and four humanoid corridor variants. |
| 4 | **MGA** | Twelve surface geometry/material variants, three peg-insertion variants and three humanoid contact tasks. |
| 5 | **Hardware** | 2GO humanoid execution, plus MGA rigid/curved/compliant scanning and peg insertion. |

## Files and regeneration

- `showcase.gif` is the GitHub-compatible looping animation.
- `showcase.mp4` is the 1344 × 864 video with the same 24-second camera journey.
- `showcase-poster.png` shows the complete gallery in one frame.
- `architecture.svg` is the separately editable framework diagram.

The compact source clips are checked in under `showcase_sources/v2/`.
[manifest.json](showcase_sources/v2/manifest.json) records all 41 checksums,
original source identities, crop bounds, selected time ranges and playback
normalization. It uses symbolic source roots rather than personal machine paths.
The gallery can be rebuilt without a simulator or the original paper folders:

```bash
python -m pip install Pillow numpy imageio-ffmpeg
python scripts/visualizations/build_readme_showcase.py
```

To inspect the composition before encoding:

```bash
python scripts/visualizations/build_readme_showcase.py \
  --preview-only --output /tmp/generative-gallery-preview
```

The separate `prepare_readme_media.py` script recreates the compact inputs when
original source folders are available. Its `--help` documents the source-root
overrides. It reads those folders and writes only to the requested output path.

## Presentation and release boundaries

All robot motion, diffusion trajectories and scene geometry come from the
original footage. Colors are preserved. Crops remove source titles, subtitles,
rulers, plots and empty margins; frames retain their aspect ratios. No robot
motion or scientific data is synthesized. Clips loop, and their selected time
ranges are rescaled to six seconds for simulation/planning or eight seconds for
hardware. Playback and the moving gallery camera do **not** indicate execution
latency, synchronization between experiments or real-time performance.

The gallery presents the broader research behind the framework. Its contents
are not a device-support or reproduction matrix:

- The original MDOC project includes CBS multi-robot coordination. This V1
  repository exposes the single-robot planning component; the multi-robot
  application remains in the [MDOC project](https://github.com/hhhhzl/mdoc).
- MGA's humanoid contact tasks are paper demonstrations. Humanoid walking/push
  and MGA GPU qualification remain follow-up work for this release.
- The MGA `soft_*` footage shows a rigid robot arm contacting compliant
  material. It does not introduce the excluded soft-robot or co-design systems.
- Hardware footage demonstrates the research systems used for the papers;
  installing V1 alone does not establish a qualified driver/controller setup
  for every robot shown.

Use the [recipe catalogue](../recipes/index.md),
[compatibility matrix](../reference/compatibility.md) and
[V1 evidence](../releases/v1_evidence.md) for runnable configurations and
validated paths.
