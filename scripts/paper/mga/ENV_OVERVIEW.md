# MGA environment overview: embedded layout (v4)

Each complete panel retains the manuscript image's **3:2** aspect ratio.
Detail views are embedded inside that canvas, not appended underneath it.
The main scene is scaled uniformly to make room for the insets. This is an environment
illustration, not a results or mechanism-performance plot. The manuscript
and its original figure assets are not changed.

## Outputs

`output/env_overview_v4/mga_environment_overview_v4.png` and `.pdf` are the
assembled figure. The PDF keeps text and the 2D schematics as vectors.
The companion JSON records image sources and interpretive assumptions.
Scanning and Peg Insertion use the exact original manuscript screenshots;
their backgrounds, cameras, materials and colors are preserved.
`output/env_overview_v4/assets/` supplies the bright Humanoid renders, with
only the box restored to the v1 warm Brax-reference material (`#C7B384`,
no sRGB-to-linear conversion). Other Humanoid colors remain unchanged.
Individual `scanning.png`, `peg_insert.png`,
and `push_to_line.png` exports have exactly the original pixel dimensions:
810 x 540, 510 x 340, and 558 x 372. The v3 outputs remain unchanged.

## Visual design

- **Surface Scanning:** a Panda overview with Hard, Soft, and Hybrid insets
  at the right, plus five shape-family illustrations inside the bottom edge.
  No explanatory text is printed below Soft.
  Hard and Soft share the same nominal geometry. Probe penetration/spring
  symbols depict compliance; they are not observations of a deforming mesh.
  The three Hybrid tiles are symbolic task-chart stiffness maps (stripes,
  hard center, soft center), not world-space textures. Hybrid is planar only;
  the five shapes apply to Hard/Soft. Shape tiles are representative diagrams,
  not exact reconstructions of a particular simulation heightfield.
- **Peg Insertion:** the robot/fixture view with three local top-view schematics
  embedded at the right. ID is nominal; Pose-OOD shows an offset/rotated socket
  against a dashed nominal outline; Sensing-OOD keeps the ID geometry and
  illustrates a biased wrench observation. The Pose-OOD displacement and
  angle are magnified for visibility and must not be read as measured values.
- **Humanoid Push:** a uniformly reduced main scene with three native thumbnails
  inside the right edge. The figure uses task names only: **Force Regulation**,
  **Fixed-stance Push**, and **Unjamming**; no P1/P2/P3 IDs or numeric targets.
  The renderer uses a manuscript-slate robot, a warm beige box, muted
  teal corridor walls and a light checker ground. Orange arrows denote task intent,
  not executed trajectories. There is no walking task in this comparison.

The H1 renderer changes only camera/materials/lighting and restores the
payload's goal-line visualization. Poses and corridor geometry are unchanged.
The unjamming inset uses the earliest *exported* pose. Its exact state index
is recorded in the current render metadata (state 0 for the v2 source export).

## Regeneration

From the repository root, using Python with NumPy, Matplotlib and Pillow:

```bash
python scripts/paper/mga/render_env_humanoid.py
python scripts/paper/mga/plot_environment_overview.py
```

The Humanoid renderer uses saved Brax states, local Chrome and a temporary
loopback HTTP server; it does not rerun simulation. The compositor reuses the
original arm screenshots and the generated v4 Humanoid assets. It does not
rerender either Panda environment and can be rerun alone to update the layout.

## Suggested figure caption

**Contact-rich environments and evaluation settings.** (a) Surface scanning
across rigid, compliant and spatially hybrid contact responses. The five
surface families apply to rigid/compliant contact; hybrid contact uses three
planar stiffness maps. (b) Peg insertion under nominal geometry and sensing
(ID), shifted socket pose (Pose-OOD), and biased wrench observations
(Sensing-OOD). Local schematics distinguish the sources of mismatch; pose
offsets are magnified. (c) Humanoid force regulation, fixed-stance pushing,
and corridor unjamming. Orange arrows indicate task objectives.

New annotations and schematics use the manuscript palette: teal `#1F7A78`,
indigo `#4B2E83`, orange `#D9660E`, slate `#3A4655`, and pale companion fills.
The original Scanning and Peg Insertion screenshots are not recolored.
